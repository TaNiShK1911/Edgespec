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

def export_model(state_dict, config_name, output_path, quantized_layers=None):
    """Export a model state dict to the EdgeSpec binary format.

    Parameters
    ----------
    state_dict : dict
        Model state dict (may contain QLinear dicts or regular tensors).
    config_name : str
        Key in OPT_CONFIGS, e.g. "opt-1.3b".
    output_path : str
        Output file path.
    quantized_layers : dict or None
        {layer_param_name: QLinear} for quantized linear layers.
        If None, treats everything as FP16.
    """
    config = OPT_CONFIGS[config_name]
    tensors_meta = []
    tensor_data = []
    current_offset = 0

    # Collect all tensors
    for name, tensor in sorted(state_dict.items()):
        if isinstance(tensor, torch.Tensor):
            data = tensor.cpu().contiguous().numpy().tobytes()
            meta = {
                "name": name,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "offset": current_offset,
                "nbytes": len(data),
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
    header_len = struct.unpack('<Q', f.read(8))[0]
    header_bytes = f.read(header_len)
    header = json.loads(header_bytes.decode('utf-8'))
    data_start = 8 + header_len
    return header, data_start


def load_tensor_by_name(filepath, tensor_name, device='cpu'):
    """Load a single tensor from the export file (low peak RAM).

    Parameters
    ----------
    filepath : str
    tensor_name : str
    device : str

    Returns
    -------
    torch.Tensor
    """
    with open(filepath, 'rb') as f:
        header, data_start = _read_header(f)

        for meta in header['tensors']:
            if meta['name'] == tensor_name:
                f.seek(data_start + meta['offset'])
                raw = f.read(meta['nbytes'])
                dtype = _DTYPE_MAP[meta['dtype']]
                tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
                tensor = tensor.reshape(meta['shape'])
                return tensor.to(device)

    raise KeyError("Tensor '%s' not found in %s" % (tensor_name, filepath))


def load_model(filepath, device='cuda', dtype=torch.float16, quantized=True):
    """Load an OPT model tensor-by-tensor from the EdgeSpec export format.

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
    with open(filepath, 'rb') as f:
        header, data_start = _read_header(f)

    config_name = header['model_name']
    config = header.get('config', OPT_CONFIGS.get(config_name, {}))

    # Build the model skeleton (empty weights)
    model = OPTModel(config, dtype=dtype)

    # Load tensors one by one
    with open(filepath, 'rb') as f:
        header, data_start = _read_header(f)

        for meta in header['tensors']:
            name = meta['name']
            f.seek(data_start + meta['offset'])
            raw = f.read(meta['nbytes'])
            tensor_dtype = _DTYPE_MAP[meta['dtype']]
            tensor = torch.frombuffer(bytearray(raw), dtype=tensor_dtype)
            tensor = tensor.reshape(meta['shape']).to(device)

            # Map tensor name to the model
            _assign_tensor(model, name, tensor, quantized=quantized)

    # Tie LM head
    model.lm_head_weight = model.embed_tokens.weight

    model = model.to(device)
    model.eval()

    print("Loaded %s from %s (%s, quantized=%s)" % (
        config_name, filepath, device, quantized))

    return model


def _assign_tensor(model, name, tensor, quantized=False):
    """Assign a loaded tensor to the appropriate place in the model.

    Handles the mapping from flat state_dict names to our model structure.
    """
    parts = name.split('.')

    # Navigate to the target module
    obj = model
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return  # skip unknown tensors

    param_name = parts[-1]

    # Handle nn.Module parameters
    if isinstance(obj, torch.nn.Module):
        if hasattr(obj, param_name):
            param = getattr(obj, param_name)
            if isinstance(param, torch.nn.Parameter):
                param.data.copy_(tensor)
            elif isinstance(param, torch.Tensor):
                setattr(obj, param_name, tensor)
            else:
                setattr(obj, param_name, tensor)
        else:
            setattr(obj, param_name, tensor)
    else:
        setattr(obj, param_name, tensor)


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
    for name, tensor in our_sd.items():
        _assign_tensor(model, name, tensor.to(dtype))

    # Tie LM head
    model.lm_head_weight = model.embed_tokens.weight

    model = model.to(device)
    model.eval()

    del hf_model, hf_sd
    if device == 'cuda':
        torch.cuda.empty_cache()

    print("Loaded %s from HuggingFace (%s)" % (config_name, device))
    return model
