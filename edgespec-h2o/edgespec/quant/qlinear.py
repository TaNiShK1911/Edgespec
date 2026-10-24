# -*- coding: utf-8 -*-
"""INT4 pack/unpack and dequantization for GPTQ weights.

Python-3.6 safe.  Reference (non-fused) path: unpack int4 from uint8,
apply scales/zeros, produce FP16 matrix for torch.matmul.

Pack format
-----------
Each uint8 byte stores two 4-bit weights: low nibble = w[2i], high nibble = w[2i+1].
``qweight`` shape: [out_features, in_features // 2]  (uint8)
``scales``  shape: [out_features, n_groups]            (float16)
``zeros``   shape: [out_features, n_groups]            (float16)

Group-quantised: each group of ``group_size`` input features shares one scale/zero.
"""

import torch
import math


class QLinear(object):
    """Quantized linear layer (INT4 GPTQ, reference dequant path).

    Parameters
    ----------
    qweight : torch.Tensor  [out_features, in_features // 2] uint8
    scales  : torch.Tensor  [out_features, n_groups] float16
    zeros   : torch.Tensor  [out_features, n_groups] float16
    bias    : torch.Tensor or None  [out_features] float16
    group_size : int
    bits : int  (always 4 for now)
    """

    def __init__(self, qweight, scales, zeros, bias=None, group_size=128, bits=4):
        self.qweight = qweight  # uint8, device may be CPU or CUDA
        self.scales = scales
        self.zeros = zeros
        self.bias = bias
        self.group_size = group_size
        self.bits = bits

        self.out_features = qweight.shape[0]
        self.in_features = qweight.shape[1] * (8 // bits)

    def dequantize(self):
        """Unpack and dequantize to a full FP16 weight matrix.

        Returns
        -------
        torch.Tensor [out_features, in_features] float16
        """
        return _dequantize_int4(
            self.qweight, self.scales, self.zeros,
            self.group_size, self.in_features
        )

    def forward(self, x):
        """y = x @ W^T + bias  where W is dequantized on-the-fly.

        Parameters
        ----------
        x : torch.Tensor [..., in_features]  float16

        Returns
        -------
        torch.Tensor [..., out_features]  float16

        Notes
        -----
        Dequantizes the full weight matrix each time.  This is the reference
        path; a fused kernel (P1-10) would avoid materialising W.
        """
        weight_fp16 = self.dequantize()
        out = torch.matmul(x, weight_fp16.t())
        if self.bias is not None:
            out = out + self.bias
        return out

    def to(self, device):
        """Move all tensors to *device*."""
        self.qweight = self.qweight.to(device)
        self.scales = self.scales.to(device)
        self.zeros = self.zeros.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        return self

    @property
    def device(self):
        return self.qweight.device


def _dequantize_int4(qweight, scales, zeros, group_size, in_features):
    """Core INT4 → FP16 dequantization.

    Parameters
    ----------
    qweight : torch.Tensor [out_features, in_features // 2] uint8
    scales  : torch.Tensor [out_features, n_groups] float16
    zeros   : torch.Tensor [out_features, n_groups] float16
    group_size : int
    in_features : int

    Returns
    -------
    torch.Tensor [out_features, in_features] float16
    """
    out_features = qweight.shape[0]

    # Unpack: each byte → two 4-bit values
    low = (qweight & 0x0F).to(torch.int8)    # even indices
    high = ((qweight >> 4) & 0x0F).to(torch.int8)  # odd indices

    # Interleave: [out, in_features]
    # low[:, i] → col 2*i, high[:, i] → col 2*i+1
    unpacked = torch.stack([low, high], dim=2).reshape(out_features, in_features)

    # Convert to float16 for dequant math
    unpacked = unpacked.to(torch.float16)

    # Per-group dequantize: weight = (int_val - zero) * scale
    n_groups = scales.shape[1]
    # Reshape to [out, n_groups, group_size]
    unpacked = unpacked.reshape(out_features, n_groups, group_size)
    # Broadcast scales and zeros: [out, n_groups, 1]
    scales_expanded = scales.unsqueeze(2)
    zeros_expanded = zeros.unsqueeze(2)

    dequantized = (unpacked - zeros_expanded) * scales_expanded
    dequantized = dequantized.reshape(out_features, in_features)

    return dequantized


def pack_int4(weight_fp, scales, zeros, group_size=128):
    """Quantize and pack an FP weight matrix to INT4 format.

    Used by the offline quantizer.

    Parameters
    ----------
    weight_fp : torch.Tensor [out_features, in_features]
    scales    : torch.Tensor [out_features, n_groups]
    zeros     : torch.Tensor [out_features, n_groups]
    group_size : int

    Returns
    -------
    qweight : torch.Tensor [out_features, in_features // 2] uint8
    """
    out_features, in_features = weight_fp.shape
    assert in_features % group_size == 0, "in_features must be divisible by group_size"
    n_groups = in_features // group_size

    weight_fp = weight_fp.reshape(out_features, n_groups, group_size)
    scales_expanded = scales.unsqueeze(2)
    zeros_expanded = zeros.unsqueeze(2)

    # Quantize: int_val = round(weight / scale + zero)
    int_val = torch.round(weight_fp / (scales_expanded + 1e-10) + zeros_expanded)
    int_val = int_val.clamp(0, 15).to(torch.uint8)
    int_val = int_val.reshape(out_features, in_features)

    # Pack two 4-bit values per byte
    assert in_features % 2 == 0
    low = int_val[:, 0::2]   # even columns
    high = int_val[:, 1::2]  # odd columns
    packed = low | (high << 4)

    return packed.to(torch.uint8)


def compute_scales_zeros(weight, group_size=128, bits=4):
    """Compute per-group scales and zeros for asymmetric quantization.

    Parameters
    ----------
    weight : torch.Tensor [out_features, in_features]
    group_size : int
    bits : int

    Returns
    -------
    scales : torch.Tensor [out_features, n_groups]
    zeros  : torch.Tensor [out_features, n_groups]
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    n_groups = in_features // group_size
    maxq = 2 ** bits - 1

    weight_groups = weight.reshape(out_features, n_groups, group_size)
    w_min = weight_groups.min(dim=2)[0]  # [out, n_groups]
    w_max = weight_groups.max(dim=2)[0]

    scales = (w_max - w_min) / maxq
    scales = scales.clamp(min=1e-10)
    zeros = -w_min / scales  # zero-point in quantized domain

    return scales.to(torch.float16), zeros.to(torch.float16)


def verify_pack_unpack(weight_fp, group_size=128, bits=4):
    """Round-trip test: quantize then dequantize, check quantization error.

    The error metric is the max absolute error normalised by the per-group
    scale (i.e. how many quantisation steps off we are).  A perfectly
    rounded value should have normalised error <= 0.5.

    Returns
    -------
    normalised_error : float
        Max abs error / scale.  Should be <= 0.5 + eps for a correct round-trip.
    passed : bool
    """
    scales, zeros = compute_scales_zeros(weight_fp, group_size, bits)
    packed = pack_int4(weight_fp.to(torch.float16), scales, zeros, group_size)
    recovered = _dequantize_int4(packed, scales, zeros, group_size, weight_fp.shape[1])

    abs_diff = (weight_fp.to(torch.float16) - recovered).abs()

    # Normalise by the per-group scale so we measure "steps off"
    out_features = weight_fp.shape[0]
    in_features = weight_fp.shape[1]
    n_groups = in_features // group_size
    abs_diff_grouped = abs_diff.reshape(out_features, n_groups, group_size)
    scales_expanded = scales.unsqueeze(2).clamp(min=1e-10)
    normalised = abs_diff_grouped / scales_expanded
    norm_error = normalised.max().item()

    # A correctly rounded value should be within 0.5 steps; allow 0.6 for FP16 noise
    return norm_error, norm_error <= 0.6
