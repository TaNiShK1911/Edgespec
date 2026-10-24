# -*- coding: utf-8 -*-
"""Low-peak-RAM model loader for EdgeSpec-H2O.

Python-3.6 safe.  Loads a custom export format tensor-by-tensor to avoid
materialising the full checkpoint in CPU memory (critical for the Nano's
3.9 GB unified RAM).

Export format
-------------
A single binary file with a JSON header followed by tensor data.

Header (JSON, length-prefixed):
  {
    "format": "edgespec-v1",
    "model_name": "opt-1.3b",
    "config": { ... },
    "tensors": [
      { "name": "layers.0.self_attn.q_proj.qweight",
        "dtype": "uint8", "shape": [2048, 1024],
        "offset": 1234, "nbytes": 2097152 },
      ...
    ]
  }

Usage:
  model = load_model("weights/opt-1.3b-4bit.edgespec", device="cuda")
"""

import io
import json
import os
import struct

import torch

from edgespec.config import OPT_CONFIGS
from edgespec.model.opt import OPTModel
from edgespec.quant.qlinear import QLinear


# --------------------------------------------------------------------------- #
#  Export (offline, on laptop)
# --------------------------------------------------------------------------- #

import hashlib

def _canonicalize_name(name):
    """Normalize tensor names to canonical format."""
    name = name.replace('.mlp.fc1.', '.fc1.').replace('.mlp.fc2.', '.fc2.')
    if name.startswith('model.decoder.'):
        name = name.replace('model.decoder.', '')
    return name


def export_model(state_dict, config_name, output_path, quantized_layers=None):
    """Export a model state dict to the EdgeSpec binary format.

    Enforces canonical naming, calculates SHA-256 checksums, and ensures
    sequential non-overlapping offsets.

    Parameters
    ----------
    state_dict : dict
        Model state dict (may contain QLinear tensors or regular tensors).
    config_name : str
        Key in OPT_CONFIGS, e.g. "opt-1.3b".
    output_path : str
        Output file path.
    quantized_layers : dict or None
        Optional dictionary of quantized layer metadata.
    """
    if config_name not in OPT_CONFIGS:
        raise ValueError("Unknown config_name: %s" % config_name)
    config = OPT_CONFIGS[config_name]
    tensors_meta = []
    tensor_data = []
    current_offset = 0

    # Collect and canonicalize all tensors
    canonical_items = []
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            cname = _canonicalize_name(name)
            canonical_items.append((cname, tensor))

    for name, tensor in sorted(canonical_items, key=lambda x: x[0]):
        data = tensor.cpu().contiguous().numpy().tobytes()
        sha = hashlib.sha256(data).hexdigest()
        meta = {
            "name": name,
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "offset": current_offset,
            "nbytes": len(data),
            "sha256": sha,
        }
        tensors_meta.append(meta)
        tensor_data.append(data)
        current_offset += len(data)

    header = {
        "format": "edgespec-v1",
        "model_name": config_name,
        "config": config,
        "tensors": tensors_meta,
    }

    header_bytes = json.dumps(header, indent=None).encode('utf-8')
    header_len = len(header_bytes)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        # 8-byte header length prefix
        f.write(struct.pack('<Q', header_len))
        f.write(header_bytes)
        for data in tensor_data:
            f.write(data)

    print("Exported %d tensors to %s (%.1f MiB)" % (
        len(tensors_meta), output_path,
        os.path.getsize(output_path) / (1024 * 1024)
    ))


# --------------------------------------------------------------------------- #
#  Load (on Nano — low peak RAM)
# --------------------------------------------------------------------------- #

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _read_header(f):
    """Read the JSON header from an EdgeSpec binary file."""
    prefix = f.read(8)
    if len(prefix) < 8:
        raise ValueError("Invalid EdgeSpec file: header prefix too short")
    header_len = struct.unpack('<Q', prefix)[0]
    header_bytes = f.read(header_len)
    if len(header_bytes) < header_len:
        raise ValueError("Invalid EdgeSpec file: truncated header bytes")
    header = json.loads(header_bytes.decode('utf-8'))
    data_start = 8 + header_len
    return header, data_start


def load_tensor_by_name(filepath, tensor_name, device='cpu'):
    """Load a single tensor from the export file (low peak RAM)."""
    with open(filepath, 'rb') as f:
        header, data_start = _read_header(f)
        c_target = _canonicalize_name(tensor_name)

        for meta in header['tensors']:
            if _canonicalize_name(meta['name']) == c_target or meta['name'] == tensor_name:
                f.seek(data_start + meta['offset'])
                raw = f.read(meta['nbytes'])
                if 'sha256' in meta:
                    actual_sha = hashlib.sha256(raw).hexdigest()
                    if actual_sha != meta['sha256']:
                        raise ValueError("Checksum mismatch for %s" % tensor_name)
                dtype = _DTYPE_MAP[meta['dtype']]
                tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
                tensor = tensor.reshape(meta['shape'])
                return tensor.to(device)

    raise KeyError("Tensor '%s' not found in %s" % (tensor_name, filepath))


def load_model(filepath, device='cuda', dtype=torch.float16, quantized=True):
    """Load an OPT model tensor-by-tensor from the EdgeSpec export format.

    Builds QLinear layers for quantized checkpoints and fails closed on
    any missing, corrupted, or unknown tensors.

    Parameters
    ----------
    filepath : str
        Path to the .edgespec file.
    device : str
        Target device ('cuda' or 'cpu').
    dtype : torch.dtype
        Model dtype (float16).
    quantized : bool
        If True, linear layers are loaded as QLinear (INT4).

    Returns
    -------
    OPTModel
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError("Checkpoint file not found: %s" % filepath)

    file_size = os.path.getsize(filepath)

    with open(filepath, 'rb') as f:
        header, data_start = _read_header(f)

    if header.get('format') != 'edgespec-v1':
        raise ValueError("Invalid checkpoint format: expected 'edgespec-v1', got %r" % header.get('format'))

    config_name = header['model_name']
    if config_name not in OPT_CONFIGS:
        raise ValueError("Unknown model config: %s" % config_name)
    config = header.get('config', OPT_CONFIGS[config_name])

    tensors_meta = header.get('tensors', [])
    if not tensors_meta:
        raise ValueError("Empty tensor manifest in %s" % filepath)

    # Validate offsets and bounds
    for meta in tensors_meta:
        off = meta['offset']
        nb = meta['nbytes']
        if data_start + off + nb > file_size:
            raise ValueError("Tensor %s bounds [%d, %d] exceed file size %d" % (
                meta['name'], data_start + off, data_start + off + nb, file_size))

    # Build alias map
    alias_map = {}
    for meta in tensors_meta:
        alias_map[meta['name']] = meta
        alias_map[_canonicalize_name(meta['name'])] = meta

    def read_tensor_from_meta(meta):
        with open(filepath, 'rb') as f_in:
            f_in.seek(data_start + meta['offset'])
            raw = f_in.read(meta['nbytes'])
        if 'sha256' in meta:
            actual_sha = hashlib.sha256(raw).hexdigest()
            if actual_sha != meta['sha256']:
                raise ValueError("Checksum mismatch for %s: expected %s, got %s" % (
                    meta['name'], meta['sha256'], actual_sha))
        tdtype = _DTYPE_MAP[meta['dtype']]
        t = torch.frombuffer(bytearray(raw), dtype=tdtype).reshape(meta['shape'])
        return t

    # Build the model skeleton
    model = OPTModel(config, dtype=dtype)
    n_layers = config['n_layers']

    # 1. Embeddings
    for emb_name in ['embed_tokens.weight', 'embed_positions.weight']:
        if emb_name not in alias_map:
            raise KeyError("Missing mandatory embedding tensor: %s" % emb_name)
        t = read_tensor_from_meta(alias_map[emb_name]).to(device).to(dtype)
        if emb_name == 'embed_tokens.weight':
            model.embed_tokens.weight.data.copy_(t)
        else:
            model.embed_positions.weight.data.copy_(t)

    # Final layer norm (decoder-level)
    if model.final_layer_norm is not None:
        if 'final_layer_norm.weight' in alias_map:
            model.final_layer_norm.weight.data.copy_(
                read_tensor_from_meta(alias_map['final_layer_norm.weight']).to(device).to(dtype)
            )
        if 'final_layer_norm.bias' in alias_map:
            model.final_layer_norm.bias.data.copy_(
                read_tensor_from_meta(alias_map['final_layer_norm.bias']).to(device).to(dtype)
            )

    # 2. Layers
    linear_subpaths = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj', 'fc1', 'fc2']

    for l_idx in range(n_layers):
        layer = model.layers[l_idx]

        # Layer norms
        for ln_name in ['self_attn_layer_norm', 'final_layer_norm']:
            w_key = 'layers.%d.%s.weight' % (l_idx, ln_name)
            b_key = 'layers.%d.%s.bias' % (l_idx, ln_name)
            ln_mod = getattr(layer, ln_name)
            if w_key in alias_map:
                ln_mod.weight.data.copy_(read_tensor_from_meta(alias_map[w_key]).to(device).to(dtype))
            if b_key in alias_map:
                ln_mod.bias.data.copy_(read_tensor_from_meta(alias_map[b_key]).to(device).to(dtype))

        # Linear layers
        for lin in linear_subpaths:
            prefix = 'layers.%d.%s' % (l_idx, lin)
            if quantized:
                qw_key = prefix + '.qweight'
                sc_key = prefix + '.scales'
                zr_key = prefix + '.zeros'
                bi_key = prefix + '.bias'

                if qw_key not in alias_map or sc_key not in alias_map or zr_key not in alias_map:
                    raise ValueError("Incomplete quantized linear layer for %s: missing qweight/scales/zeros" % prefix)

                qweight = read_tensor_from_meta(alias_map[qw_key]).to(device)
                scales = read_tensor_from_meta(alias_map[sc_key]).to(device)
                zeros = read_tensor_from_meta(alias_map[zr_key]).to(device)
                bias = read_tensor_from_meta(alias_map[bi_key]).to(device) if bi_key in alias_map else None

                in_f = qweight.shape[1] * 2
                n_groups = scales.shape[1]
                group_size = in_f // n_groups if n_groups > 1 else -1

                ql = QLinear(qweight, scales, zeros, bias=bias, group_size=group_size, bits=4).to(device)

                if 'self_attn' in lin:
                    proj = lin.split('.')[-1]
                    setattr(layer.self_attn, proj, ql)
                elif lin == 'fc1':
                    layer.mlp.fc1 = ql
                elif lin == 'fc2':
                    layer.mlp.fc2 = ql
            else:
                w_key = prefix + '.weight'
                bi_key = prefix + '.bias'
                if w_key not in alias_map:
                    raise ValueError("Missing weight tensor for %s" % prefix)
                w = read_tensor_from_meta(alias_map[w_key]).to(device).to(dtype)
                bias = read_tensor_from_meta(alias_map[bi_key]).to(device).to(dtype) if bi_key in alias_map else None

                if 'self_attn' in lin:
                    proj = lin.split('.')[-1]
                    mod = getattr(layer.self_attn, proj)
                    mod.weight.data.copy_(w)
                    if bias is not None and mod.bias is not None:
                        mod.bias.data.copy_(bias)
                elif lin == 'fc1':
                    layer.fc1.weight.data.copy_(w)
                    if bias is not None and layer.fc1.bias is not None:
                        layer.fc1.bias.data.copy_(bias)
                elif lin == 'fc2':
                    layer.fc2.weight.data.copy_(w)
                    if bias is not None and layer.fc2.bias is not None:
                        layer.fc2.bias.data.copy_(bias)

    # 3. Strict validation: check for unknown tensors in checkpoint
    known_roots = {'embed_tokens', 'embed_positions', 'final_layer_norm', 'lm_head', 'lm_head_weight'}
    for meta in tensors_meta:
        cname = _canonicalize_name(meta['name'])
        root = cname.split('.')[0]
        if root not in known_roots and root != 'layers':
            raise KeyError("Unknown tensor '%s' in checkpoint" % meta['name'])
        if root == 'layers':
            parts = cname.split('.')
            if len(parts) < 3 or not parts[1].isdigit() or int(parts[1]) >= n_layers:
                raise KeyError("Unknown or out-of-range layer tensor '%s'" % meta['name'])

    # Tie LM head
    model.lm_head_weight = model.embed_tokens.weight
    model = model.to(device)
    model.eval()

    print("Loaded %s from %s (%s, quantized=%s)" % (
        config_name, filepath, device, quantized))
    return model


def load_model_from_hf(model_name, device='cuda', dtype=torch.float16):
    """Load an OPT model from HuggingFace weights into our vendored model.

    This is used on the laptop for development and golden-file generation.
    Requires ``transformers`` to be installed.

    Parameters
    ----------
    model_name : str
        HuggingFace model id, e.g. "facebook/opt-125m".
    device : str
    dtype : torch.dtype

    Returns
    -------
    OPTModel
    """
    from transformers import AutoModelForCausalLM

    # Determine config
    if '125m' in model_name.lower():
        config_name = 'opt-125m'
    elif '350m' in model_name.lower():
        config_name = 'opt-350m'
    elif '1.3b' in model_name.lower():
        config_name = 'opt-1.3b'
    else:
        raise ValueError("Unknown model: %s" % model_name)

    config = OPT_CONFIGS[config_name]

    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    )
    hf_sd = hf_model.state_dict()

    # Build our model
    model = OPTModel(config, dtype=dtype)

    # Map HF state dict keys to our model
    our_sd = {}

    # Embeddings
    our_sd['embed_tokens.weight'] = hf_sd['model.decoder.embed_tokens.weight']
    our_sd['embed_positions.weight'] = hf_sd['model.decoder.embed_positions.weight']

    # Final layer norm (pre-LN models)
    if 'model.decoder.final_layer_norm.weight' in hf_sd:
        our_sd['final_layer_norm.weight'] = hf_sd['model.decoder.final_layer_norm.weight']
        our_sd['final_layer_norm.bias'] = hf_sd['model.decoder.final_layer_norm.bias']

    # Layers
    for i in range(config['n_layers']):
        prefix_hf = 'model.decoder.layers.%d' % i
        prefix_ours = 'layers.%d' % i

        # Self attention
        for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            our_sd['%s.self_attn.%s.weight' % (prefix_ours, proj)] = \
                hf_sd['%s.self_attn.%s.weight' % (prefix_hf, proj)]
            our_sd['%s.self_attn.%s.bias' % (prefix_ours, proj)] = \
                hf_sd['%s.self_attn.%s.bias' % (prefix_hf, proj)]

        # Layer norms
        our_sd['%s.self_attn_layer_norm.weight' % prefix_ours] = \
            hf_sd['%s.self_attn_layer_norm.weight' % prefix_hf]
        our_sd['%s.self_attn_layer_norm.bias' % prefix_ours] = \
            hf_sd['%s.self_attn_layer_norm.bias' % prefix_hf]
        our_sd['%s.final_layer_norm.weight' % prefix_ours] = \
            hf_sd['%s.final_layer_norm.weight' % prefix_hf]
        our_sd['%s.final_layer_norm.bias' % prefix_ours] = \
            hf_sd['%s.final_layer_norm.bias' % prefix_hf]

        # MLP
        our_sd['%s.mlp.fc1.weight' % prefix_ours] = \
            hf_sd['%s.fc1.weight' % prefix_hf]
        our_sd['%s.mlp.fc1.bias' % prefix_ours] = \
            hf_sd['%s.fc1.bias' % prefix_hf]
        our_sd['%s.mlp.fc2.weight' % prefix_ours] = \
            hf_sd['%s.fc2.weight' % prefix_hf]
        our_sd['%s.mlp.fc2.bias' % prefix_ours] = \
            hf_sd['%s.fc2.bias' % prefix_hf]

    # Load state dict into our model
    model.load_state_dict({k: v.to(dtype) for k, v in our_sd.items()})

    # Tie LM head
    model.lm_head_weight = model.embed_tokens.weight

    model = model.to(device)
    model.eval()

    del hf_model, hf_sd
    if device == 'cuda':
        torch.cuda.empty_cache()

    print("Loaded %s from HuggingFace (%s)" % (config_name, device))
    return model
