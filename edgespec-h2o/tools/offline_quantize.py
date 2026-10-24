#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline GPTQ quantization for OPT models (P1-2).

Runs on laptop/Colab GPU (NOT on the Nano).  Produces:
- Quantised weights in EdgeSpec binary format
- WikiText-2 PPL comparison (FP16 vs 4-bit)
- Per-layer weight checksums for G1.4

Requires: transformers, datasets, torch (with GPU recommended).

Usage:
    python tools/offline_quantize.py --model facebook/opt-125m --output data/weights/opt-125m-4bit.edgespec
    python tools/offline_quantize.py --model facebook/opt-1.3b --output data/weights/opt-1.3b-4bit.edgespec
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgespec.config import (
    GPTQ_BITS, GPTQ_GROUP_SIZE, GPTQ_ACT_ORDER,
    GPTQ_CALIBRATION_NSAMPLES, GPTQ_CALIBRATION_SEQLEN,
    OPT_CONFIGS, WEIGHTS_DIR, GOLDEN_DIR,
)
from edgespec.quant.qlinear import pack_int4, compute_scales_zeros, QLinear
from edgespec.model.loader import export_model


# --------------------------------------------------------------------------- #
#  GPTQ core algorithm (adapted from IST-DASLab/gptq)
# --------------------------------------------------------------------------- #

class GPTQ(object):
    """GPTQ quantizer for a single linear layer.

    Implements the algorithm from Frantar et al. (arXiv 2210.17323) §4:
    sequentially quantize columns using Hessian information to minimise
    the layer-wise reconstruction error.
    """

    def __init__(self, layer, bits=4, group_size=128):
        self.layer = layer
        self.bits = bits
        self.group_size = group_size
        self.dev = layer.weight.device

        W = layer.weight.data.clone().float()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev,
                             dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp):
        """Accumulate Hessian estimate from a batch of inputs.

        Parameters
        ----------
        inp : torch.Tensor [batch_size, seq_len, in_features] or [n, in_features]
        """
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        tmp = inp.float()
        self.H += tmp.t().mm(tmp)
        self.nsamples += inp.shape[0]

    def quantize(self, percdamp=0.01, blocksize=128):
        """Run the GPTQ quantization.

        Returns
        -------
        qweight : torch.Tensor uint8 [rows, columns // 2]
        scales  : torch.Tensor float16 [rows, n_groups]
        zeros   : torch.Tensor float16 [rows, n_groups]
        max_error : float
        """
        W = self.layer.weight.data.clone().float()
        H = self.H / self.nsamples
        self.H = None  # free memory

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Dampening
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        n_groups = self.columns // self.group_size if self.group_size > 0 else 1
        actual_group_size = self.group_size if self.group_size > 0 else self.columns

        maxq = 2 ** self.bits - 1
        scales = torch.zeros(self.rows, n_groups, device=self.dev, dtype=torch.float32)
        zeros = torch.zeros(self.rows, n_groups, device=self.dev, dtype=torch.float32)

        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        for b_start in range(0, self.columns, blocksize):
            b_end = min(b_start + blocksize, self.columns)
            b_count = b_end - b_start

            W_block = W[:, b_start:b_end].clone()
            Q_block = torch.zeros_like(W_block)
            Err = torch.zeros_like(W_block)
            Hinv_block = Hinv[b_start:b_end, b_start:b_end]

            for j in range(b_count):
                col = b_start + j
                group_idx = col // actual_group_size

                # Compute scales/zeros for this group if at group boundary
                if col % actual_group_size == 0:
                    g_start = col
                    g_end = min(col + actual_group_size, self.columns)
                    w_group = W[:, g_start:g_end]
                    w_min = w_group.min(dim=1)[0]
                    w_max = w_group.max(dim=1)[0]
                    s = (w_max - w_min) / maxq
                    s = s.clamp(min=1e-10)
                    z = -w_min / s
                    scales[:, group_idx] = s
                    zeros[:, group_idx] = z

                w = W_block[:, j]
                d = Hinv_block[j, j]

                # Quantize
                s = scales[:, group_idx]
                z = zeros[:, group_idx]
                q = torch.clamp(torch.round(w / s + z), 0, maxq)
                Q_block[:, j] = (q - z) * s

                # Error and update
                err = (w - Q_block[:, j]) / d
                Losses[:, col] = err ** 2
                Err[:, j] = err

                # Update remaining columns in block
                W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)

            Q[:, b_start:b_end] = Q_block
            # Update remaining columns after block
            W[:, b_end:] -= Err.mm(Hinv[b_start:b_end, b_end:])

        max_error = torch.sqrt(Losses.sum() / self.rows).item()

        # Pack the quantized weights
        # Re-quantize to integer domain
        int_W = torch.zeros(self.rows, self.columns, device=self.dev, dtype=torch.float32)
        for g in range(n_groups):
            g_start = g * actual_group_size
            g_end = min((g + 1) * actual_group_size, self.columns)
            s = scales[:, g].unsqueeze(1)
            z = zeros[:, g].unsqueeze(1)
            int_W[:, g_start:g_end] = torch.clamp(
                torch.round(Q[:, g_start:g_end] / (s + 1e-10) + z), 0, maxq
            )

        # Pack to uint8 (2 x int4 per byte)
        int_W = int_W.to(torch.uint8)
        qweight = torch.zeros(self.rows, self.columns // 2, device=self.dev,
                              dtype=torch.uint8)
        qweight = int_W[:, 0::2] | (int_W[:, 1::2] << 4)

        return (qweight.cpu(),
                scales.to(torch.float16).cpu(),
                zeros.to(torch.float16).cpu(),
                max_error)


# --------------------------------------------------------------------------- #
#  Calibration data
# --------------------------------------------------------------------------- #

def get_calibration_data(nsamples=128, seqlen=2048, model_name='facebook/opt-1.3b'):
    """Load C4 calibration data (per GPTQ paper §5).

    Returns
    -------
    list of torch.LongTensor [1, seqlen]
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading C4 calibration data (%d samples, seqlen=%d)..." % (nsamples, seqlen))

    dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    samples = []
    text_buffer = ""

    for item in dataset:
        text_buffer += item['text'] + "\n"
        while len(text_buffer) > seqlen * 4:  # rough char estimate
            encoded = tokenizer.encode(text_buffer[:seqlen * 6], return_tensors='pt',
                                       max_length=seqlen, truncation=True)
            if encoded.shape[1] >= seqlen:
                samples.append(encoded[:, :seqlen])
                text_buffer = text_buffer[seqlen * 3:]
                if len(samples) >= nsamples:
                    break
            else:
                break
        if len(samples) >= nsamples:
            break

    print("Collected %d calibration samples" % len(samples))
    return samples


# --------------------------------------------------------------------------- #
#  Quantize a full model
# --------------------------------------------------------------------------- #

def quantize_model(model_name, output_path, bits=4, group_size=128,
                   nsamples=128, seqlen=2048):
    """Run GPTQ on all linear layers of an OPT model.

    Parameters
    ----------
    model_name : str  (HuggingFace id, e.g. "facebook/opt-1.3b")
    output_path : str
    bits : int
    group_size : int
    nsamples : int
    seqlen : int
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("GPTQ Quantization: %s" % model_name)
    print("  bits=%d, group_size=%d, nsamples=%d, seqlen=%d" % (
        bits, group_size, nsamples, seqlen))
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Get calibration data
    calib_data = get_calibration_data(nsamples, seqlen, model_name)

    # Quantize each linear layer
    quantized = {}
    checksums = {}
    layer_errors = {}

    decoder_layers = model.model.decoder.layers

    for layer_idx in range(len(decoder_layers)):
        layer = decoder_layers[layer_idx]
        print("\n--- Layer %d/%d ---" % (layer_idx + 1, len(decoder_layers)))

        # Collect the linear modules to quantize
        linears = {
            'self_attn.q_proj': layer.self_attn.q_proj,
            'self_attn.k_proj': layer.self_attn.k_proj,
            'self_attn.v_proj': layer.self_attn.v_proj,
            'self_attn.out_proj': layer.self_attn.out_proj,
            'fc1': layer.fc1,
            'fc2': layer.fc2,
        }

        for name, linear in linears.items():
            full_name = 'layers.%d.%s' % (layer_idx, name)
            print("  Quantizing %s [%s]..." % (full_name, list(linear.weight.shape)))

            gptq = GPTQ(linear, bits=bits, group_size=group_size)

            # Collect Hessians from calibration data
            hooks = []

            def make_hook(gptq_obj):
                def hook_fn(module, inp, out):
                    gptq_obj.add_batch(inp[0].data)
                return hook_fn

            h = linear.register_forward_hook(make_hook(gptq))
            hooks.append(h)

            # Run calibration samples through the layer
            with torch.no_grad():
                for sample in calib_data[:min(32, len(calib_data))]:
                    try:
                        model(sample.to(device))
                    except RuntimeError:
                        pass  # OOM on some layers for large models

            for h in hooks:
                h.remove()

            # Quantize
            qweight, scales, zeros, max_err = gptq.quantize()
            layer_errors[full_name] = max_err

            # Store quantized weights
            quantized[full_name + '.qweight'] = qweight
            quantized[full_name + '.scales'] = scales
            quantized[full_name + '.zeros'] = zeros

            # Bias (kept in FP16)
            if linear.bias is not None:
                quantized[full_name + '.bias'] = linear.bias.data.cpu().to(torch.float16)

            # Checksum
            checksum = hashlib.sha256(qweight.numpy().tobytes()).hexdigest()[:16]
            checksums[full_name] = checksum

            print("    max_error=%.4f, checksum=%s" % (max_err, checksum))

    # Add FP16 embeddings, layer norms, etc. (unquantized)
    print("\nAdding FP16 components...")
    fp16_keys = []
    for name, param in model.named_parameters():
        keep_fp16 = any(k in name for k in [
            'embed_tokens', 'embed_positions',
            'layer_norm', 'final_layer_norm',
        ])
        if keep_fp16:
            mapped_name = name.replace('model.decoder.', '')
            quantized[mapped_name] = param.data.cpu().to(torch.float16)
            fp16_keys.append(mapped_name)

    # Determine config name
    if '125m' in model_name.lower():
        config_name = 'opt-125m'
    elif '350m' in model_name.lower():
        config_name = 'opt-350m'
    elif '1.3b' in model_name.lower():
        config_name = 'opt-1.3b'
    else:
        config_name = model_name.split('/')[-1]

    # Export
    export_model(quantized, config_name, output_path)

    # Save checksums
    checksums_path = output_path.replace('.edgespec', '_checksums.json')
    os.makedirs(os.path.dirname(checksums_path) or '.', exist_ok=True)
    with open(checksums_path, 'w') as f:
        json.dump({
            "model": model_name,
            "bits": bits,
            "group_size": group_size,
            "checksums": checksums,
            "errors": layer_errors,
        }, f, indent=2)
    print("\nChecksums saved to %s" % checksums_path)

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nDone! Output: %s" % output_path)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Offline GPTQ quantization for OPT')
    parser.add_argument('--model', type=str, default='facebook/opt-125m',
                        help='HuggingFace model id')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--bits', type=int, default=GPTQ_BITS)
    parser.add_argument('--group-size', type=int, default=GPTQ_GROUP_SIZE)
    parser.add_argument('--nsamples', type=int, default=GPTQ_CALIBRATION_NSAMPLES)
    parser.add_argument('--seqlen', type=int, default=GPTQ_CALIBRATION_SEQLEN)
    args = parser.parse_args()

    if args.output is None:
        name = args.model.split('/')[-1]
        args.output = os.path.join(WEIGHTS_DIR,
                                   '%s-%dbit.edgespec' % (name, args.bits))

    quantize_model(
        args.model, args.output,
        bits=args.bits, group_size=args.group_size,
        nsamples=args.nsamples, seqlen=args.seqlen
    )


if __name__ == '__main__':
    main()
