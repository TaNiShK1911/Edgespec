# -*- coding: utf-8 -*-
"""Tests for QLinear INT4 pack/unpack (G1.4)."""

import os
import sys
import pytest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgespec.quant.qlinear import (
    QLinear, pack_int4, compute_scales_zeros,
    _dequantize_int4, verify_pack_unpack
)


class TestPackUnpack(object):
    """INT4 pack/unpack round-trip tests."""

    def test_roundtrip_small(self):
        """Pack then unpack a small matrix — normalised error should be <= 0.6."""
        torch.manual_seed(42)
        weight = torch.randn(64, 128, dtype=torch.float32)
        norm_err, passed = verify_pack_unpack(weight, group_size=128, bits=4)
        print("Small matrix normalised error: %.6f" % norm_err)
        assert passed, "Normalised error %.6f > 0.6" % norm_err

    def test_roundtrip_medium(self):
        """Pack/unpack a medium matrix (simulating one OPT layer)."""
        torch.manual_seed(42)
        weight = torch.randn(2048, 2048, dtype=torch.float32)
        norm_err, passed = verify_pack_unpack(weight, group_size=128, bits=4)
        print("Medium matrix normalised error: %.6f" % norm_err)
        assert passed, "Normalised error %.6f > 0.6" % norm_err

    def test_unpack_values_in_range(self):
        """Unpacked INT4 values should be in [0, 15]."""
        torch.manual_seed(42)
        weight = torch.randn(64, 128)
        scales, zeros = compute_scales_zeros(weight, group_size=128)
        packed = pack_int4(weight.to(torch.float16), scales, zeros)

        # Unpack
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F

        assert low.max() <= 15
        assert low.min() >= 0
        assert high.max() <= 15
        assert high.min() >= 0

    def test_pack_shape(self):
        """Packed tensor should be half the columns."""
        weight = torch.randn(256, 512)
        scales, zeros = compute_scales_zeros(weight, group_size=128)
        packed = pack_int4(weight.to(torch.float16), scales, zeros)

        assert packed.shape == (256, 256), \
            "Expected (256, 256), got %s" % (packed.shape,)

    def test_qlinear_forward(self):
        """QLinear forward should produce correct output shape."""
        torch.manual_seed(42)
        in_f, out_f = 512, 256
        weight = torch.randn(out_f, in_f)
        scales, zeros = compute_scales_zeros(weight, group_size=128)
        packed = pack_int4(weight.to(torch.float16), scales, zeros, group_size=128)

        ql = QLinear(packed, scales, zeros, group_size=128)

        x = torch.randn(1, 8, in_f, dtype=torch.float16)
        y = ql.forward(x)

        assert y.shape == (1, 8, out_f), \
            "Expected (1, 8, %d), got %s" % (out_f, y.shape)

    def test_qlinear_accuracy(self):
        """QLinear output should be close to the FP16 reference matmul.

        4-bit quantization introduces error proportional to the weight scale;
        we check that the output correlation is high and mean-abs-error is
        bounded relative to output magnitude.
        """
        torch.manual_seed(42)
        in_f, out_f = 512, 256
        weight = torch.randn(out_f, in_f, dtype=torch.float32)
        scales, zeros = compute_scales_zeros(weight, group_size=128)
        packed = pack_int4(weight.to(torch.float16), scales, zeros, group_size=128)

        ql = QLinear(packed, scales, zeros, group_size=128)

        x = torch.randn(1, 4, in_f, dtype=torch.float16)

        # QLinear output
        y_quant = ql.forward(x)

        # FP16 reference
        y_ref = torch.matmul(x, weight.to(torch.float16).t())

        # Mean absolute error normalised by output std
        mae = (y_quant - y_ref).abs().mean().item()
        out_std = y_ref.std().item()
        normalised_mae = mae / (out_std + 1e-6)

        print("QLinear vs FP16: MAE=%.4f, out_std=%.4f, norm_MAE=%.4f" % (
            mae, out_std, normalised_mae))
        # 4-bit quant typically gives normalised MAE < 0.5
        assert normalised_mae < 0.5, "Normalised MAE %.4f too high" % normalised_mae

    def test_scales_zeros_shape(self):
        """Scales and zeros should have correct shapes."""
        weight = torch.randn(256, 512)
        scales, zeros = compute_scales_zeros(weight, group_size=128)

        assert scales.shape == (256, 4), \
            "Expected scales (256, 4), got %s" % (scales.shape,)
        assert zeros.shape == (256, 4), \
            "Expected zeros (256, 4), got %s" % (zeros.shape,)

    def test_device_transfer(self):
        """QLinear.to() should move all tensors."""
        torch.manual_seed(42)
        weight = torch.randn(64, 128)
        scales, zeros = compute_scales_zeros(weight, group_size=128)
        packed = pack_int4(weight.to(torch.float16), scales, zeros, group_size=128)
        bias = torch.randn(64, dtype=torch.float16)

        ql = QLinear(packed, scales, zeros, bias=bias, group_size=128)
        ql.to('cpu')

        assert ql.qweight.device.type == 'cpu'
        assert ql.scales.device.type == 'cpu'
        assert ql.zeros.device.type == 'cpu'
        assert ql.bias.device.type == 'cpu'
