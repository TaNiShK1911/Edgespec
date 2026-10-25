#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert an existing FP16 PyTorch checkpoint to .edgespec 4-bit INT4 format.

Uses RTN (Round-To-Nearest) group-128 asymmetric quantization with scales and zeros.
Produces an .edgespec binary file ready for QLinear execution on Jetson Nano.
"""

from __future__ import print_function

import argparse
import os
import sys
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from edgespec.config import OPT_CONFIGS
from edgespec.quant.qlinear import compute_scales_zeros, pack_int4
from edgespec.model.loader import export_model


def convert_checkpoint(model_dir, config_name, output_file, group_size=128):
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError("Missing %s" % bin_path)

    print("[+] Loading FP16 weights from %s (%.1f MiB)..." %
          (bin_path, os.path.getsize(bin_path) / (1024 * 1024)))
    t0 = time.time()
    sd = torch.load(bin_path, map_location="cpu")
    print("[+] Checkpoint loaded in %.2fs" % (time.time() - t0))

    config = OPT_CONFIGS[config_name]
    n_layers = config["n_layers"]

    import gc
    new_sd = {}
    if "model.decoder.embed_tokens.weight" in sd:
        new_sd["embed_tokens.weight"] = sd.pop("model.decoder.embed_tokens.weight").half()
    if "model.decoder.embed_positions.weight" in sd:
        new_sd["embed_positions.weight"] = sd.pop("model.decoder.embed_positions.weight").half()
    if "model.decoder.final_layer_norm.weight" in sd:
        new_sd["final_layer_norm.weight"] = sd.pop("model.decoder.final_layer_norm.weight").half()
        new_sd["final_layer_norm.bias"] = sd.pop("model.decoder.final_layer_norm.bias").half()

    linear_names = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.out_proj", "fc1", "fc2"
    ]

    print("[+] Quantizing %d layers to INT4 (group_size=%d)..." % (n_layers, group_size))
    for l_idx in range(n_layers):
        pfx = "model.decoder.layers.%d" % l_idx
        our_pfx = "layers.%d" % l_idx

        # Layer norms (remain FP16)
        for ln in ["self_attn_layer_norm", "final_layer_norm"]:
            key_w = "%s.%s.weight" % (pfx, ln)
            key_b = "%s.%s.bias" % (pfx, ln)
            if key_w in sd:
                new_sd["%s.%s.weight" % (our_pfx, ln)] = sd.pop(key_w).half()
            if key_b in sd:
                new_sd["%s.%s.bias" % (our_pfx, ln)] = sd.pop(key_b).half()

        # Linear weights -> INT4
        for lin in linear_names:
            key_w = "%s.%s.weight" % (pfx, lin)
            key_b = "%s.%s.bias" % (pfx, lin)
            if key_w in sd:
                w = sd.pop(key_w).float()
                sc, zr = compute_scales_zeros(w, group_size=group_size, bits=4)
                qw = pack_int4(w.half(), sc, zr, group_size=group_size)

                new_sd["%s.%s.qweight" % (our_pfx, lin)] = qw
                new_sd["%s.%s.scales" % (our_pfx, lin)] = sc
                new_sd["%s.%s.zeros" % (our_pfx, lin)] = zr
                del w

            if key_b in sd:
                new_sd["%s.%s.bias" % (our_pfx, lin)] = sd.pop(key_b).half()

        if (l_idx + 1) % 4 == 0 or l_idx == n_layers - 1:
            print("  [layer %2d/%2d done]" % (l_idx + 1, n_layers))
            gc.collect()

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print("[+] Exporting to %s..." % output_file)
    export_model(new_sd, config_name, output_file)
    file_size_mib = os.path.getsize(output_file) / (1024 * 1024)
    print("[+] Done! 4-bit model exported: %.1f MiB (compressed from %.1f MiB)" %
          (file_size_mib, os.path.getsize(bin_path) / (1024 * 1024)))
    return output_file


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert FP16 OPT checkpoint to INT4 .edgespec")
    p.add_argument("--model-dir", default=os.path.expanduser("~/edgespec/models/opt-125m"))
    p.add_argument("--model-size", default="125m", choices=["125m", "350m", "1.3b"])
    p.add_argument("--output", default="data/weights/opt-125m-4bit.edgespec")
    p.add_argument("--group-size", type=int, default=128)
    args = p.parse_args()

    config_name = "opt-%s" % args.model_size
    convert_checkpoint(os.path.expanduser(args.model_dir), config_name, args.output, args.group_size)
