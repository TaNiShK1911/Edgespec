#!/usr/bin/env python3
"""
EdgeSpec-H2O: Interactive / single-shot OPT inference with H2O KV Cache on Jetson Nano.

Supports:
- OPT-125M, OPT-350M, OPT-1.3B
- INT4 Quantized checkpoints (.edgespec binary format) via QLinear
- FP16/FP32 PyTorch checkpoints (pytorch_model.bin)
- H2O compressed KV cache (default 20% budget) or standard KV cache
- CUDA acceleration on NVIDIA Jetson Nano Tegra X1 (or CPU fallback)

Usage:
------
# Interactive chat with OPT-125M on CUDA using H2O Cache:
python tools/chat.py --model-dir ~/edgespec/models/opt-125m --device cuda --use-h2o

# Interactive chat with quantized OPT-1.3B (.edgespec) on CUDA using H2O Cache:
python tools/chat.py --weights data/weights/opt-1.3b-4bit.edgespec --device cuda --use-h2o

# Single prompt non-interactive:
python tools/chat.py --model-dir ~/edgespec/models/opt-125m --device cuda --prompt "Artificial intelligence is"
"""
import argparse
import os
import sys
import time
import json

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from edgespec.config import OPT_CONFIGS, H2O_DEFAULTS
from edgespec.model.opt import OPTModel
from edgespec.tokenizer import get_tokenizer
from edgespec.cache.h2o_cache import H2OCache, create_h2o_caches


def parse_args():
    p = argparse.ArgumentParser(description="EdgeSpec-H2O Inference & Chat")
    p.add_argument("--model-dir", default=os.path.expanduser("~/edgespec/models/opt-125m"),
                   help="Path to directory containing pytorch_model.bin and vocab.json")
    p.add_argument("--weights", default=None,
                   help="Path to .edgespec INT4 quantized weights file")
    p.add_argument("--model-size", default="125m", choices=["125m", "350m", "1.3b"],
                   help="Model size variant if loading from directory")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="Execution device ('cuda' or 'cpu')")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                   help="Computation dtype (float16 recommended on CUDA)")
    p.add_argument("--use-h2o", action="store_true", default=True,
                   help="Use H2O compressed KV cache (default: True)")
    p.add_argument("--h2o-budget", type=float, default=0.20,
                   help="H2O KV-cache budget fraction (default: 0.20 = 20%%)")
    p.add_argument("--decay", type=float, default=0.98,
                   help="H2O score decay factor lambda (default: 0.98)")
    p.add_argument("--max-new-tokens", type=int, default=100,
                   help="Maximum tokens to generate")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (0 for greedy)")
    p.add_argument("--top-k", type=int, default=40,
                   help="Top-k sampling pool size")
    p.add_argument("--prompt", default=None,
                   help="Single prompt (if omitted, starts interactive chat REPL)")
    return p.parse_args()


def load_tokenizer(model_dir):
    """Load BPE tokenizer from model directory or data folder."""
    vocab_file = os.path.join(model_dir, "vocab.json")
    merges_file = os.path.join(model_dir, "merges.txt")
    if not os.path.exists(vocab_file):
        # Fallback to repository data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        vocab_file = os.path.join(data_dir, "vocab.json")
        merges_file = os.path.join(data_dir, "merges.txt")

    print("[+] Loading cleanroom tokenizer from %s ..." % os.path.dirname(vocab_file))
    return get_tokenizer(vocab_file=vocab_file, merges_file=merges_file)


def load_model_instance(args):
    """Load OPT model from .edgespec (quantized INT4) or pytorch_model.bin."""
    target_device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    target_dtype = torch.float16 if (target_device == "cuda" and args.dtype == "float16") else torch.float32

    # Case 1: INT4 Quantized .edgespec file
    if args.weights and os.path.isfile(args.weights):
        from edgespec.model.loader import load_model
        print("[+] Loading INT4 quantized model from %s on %s (%s) ..." %
              (args.weights, target_device, target_dtype))
        t0 = time.time()
        model = load_model(args.weights, device=target_device, dtype=target_dtype, quantized=True)
        print("[+] INT4 model loaded in %.2fs" % (time.time() - t0))
        return model, target_device, target_dtype

    # Case 2: PyTorch checkpoint directory
    model_dir = os.path.expanduser(args.model_dir)
    cfg_file = os.path.join(model_dir, "config.json")
    config_name = "opt-%s" % args.model_size
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cfg_data = json.load(f)
        n_layers = cfg_data.get("num_hidden_layers", 12)
        for k, v in OPT_CONFIGS.items():
            if v.get("n_layers") == n_layers:
                config_name = k
                break

    config = OPT_CONFIGS[config_name]
    print("[+] Initializing OPTModel skeleton (%s, %d layers) ..." % (config_name, config["n_layers"]))
    model = OPTModel(config, dtype=target_dtype).to(target_device).to(target_dtype)

    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        print("[+] Loading weights from %s (%.1f MiB) ..." %
              (bin_path, os.path.getsize(bin_path) / (1024 * 1024)))
        t0 = time.time()
        sd = torch.load(bin_path, map_location="cpu")

        # Map state dict
        mapping = {}
        mapping["embed_tokens.weight"] = sd.get("model.decoder.embed_tokens.weight")
        mapping["embed_positions.weight"] = sd.get("model.decoder.embed_positions.weight")
        if "model.decoder.final_layer_norm.weight" in sd:
            mapping["final_layer_norm.weight"] = sd["model.decoder.final_layer_norm.weight"]
            mapping["final_layer_norm.bias"] = sd["model.decoder.final_layer_norm.bias"]

        for i in range(config["n_layers"]):
            hf_pfx = "model.decoder.layers.%d" % i
            our_pfx = "layers.%d" % i
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                mapping["%s.self_attn.%s.weight" % (our_pfx, proj)] = sd.get("%s.self_attn.%s.weight" % (hf_pfx, proj))
                mapping["%s.self_attn.%s.bias" % (our_pfx, proj)] = sd.get("%s.self_attn.%s.bias" % (hf_pfx, proj))
            for ln in ("self_attn_layer_norm", "final_layer_norm"):
                mapping["%s.%s.weight" % (our_pfx, ln)] = sd.get("%s.%s.weight" % (hf_pfx, ln))
                mapping["%s.%s.bias" % (our_pfx, ln)] = sd.get("%s.%s.bias" % (hf_pfx, ln))
            mapping["%s.mlp.fc1.weight" % our_pfx] = sd.get("%s.fc1.weight" % hf_pfx)
            mapping["%s.mlp.fc1.bias" % our_pfx] = sd.get("%s.fc1.bias" % hf_pfx)
            mapping["%s.mlp.fc2.weight" % our_pfx] = sd.get("%s.fc2.weight" % hf_pfx)
            mapping["%s.mlp.fc2.bias" % our_pfx] = sd.get("%s.fc2.bias" % hf_pfx)

        for name, tensor in mapping.items():
            if tensor is not None:
                parts = name.split('.')
                obj = model
                for p in parts[:-1]:
                    if p.isdigit():
                        obj = obj[int(p)]
                    else:
                        obj = getattr(obj, p)
                param = getattr(obj, parts[-1])
                if isinstance(param, torch.nn.Parameter):
                    param.data.copy_(tensor.to(target_device).to(target_dtype))
                else:
                    setattr(obj, parts[-1], tensor.to(target_device).to(target_dtype))

        print("[+] Checkpoint loaded in %.2fs" % (time.time() - t0))
    else:
        print("[!] No pytorch_model.bin found in %s — running with initialized weights" % model_dir)

    model.lm_head_weight = model.embed_tokens.weight
    model = model.to(target_device).to(target_dtype)
    model.eval()
    return model, target_device, target_dtype


def generate_response(model, tok, prompt_text, args, device, dtype):
    """Autoregressive text generation using H2O compressed KV Cache."""
    ids = tok.encode(prompt_text, add_bos=True)
    if len(ids) == 0:
        return ""

    sep = "=" * 65
    print("\n" + sep)
    print("Prompt  : %s" % prompt_text)
    print(sep)
    print("Output  : ", end="", flush=True)

    generated = list(ids)
    t0 = time.time()

    # Configure H2O KV Cache
    max_ctx = model.config.get("max_positions", 2048)
    budget = max(32, int(max_ctx * args.h2o_budget))
    n_local = min(128, budget // 2)
    n_pin = 4
    n_hh = max(0, budget - n_local - n_pin)
    n_scratch = 8

    if args.use_h2o:
        model.init_h2o_caches(
            n_hh=n_hh, n_local=n_local, n_pin=n_pin, n_scratch=n_scratch,
            decay=args.decay, evict_every=1, device=device
        )
    else:
        model.init_kv_caches(max_seq_len=max_ctx, device=device)

    with torch.no_grad():
        prompt_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        # 1. Prefill
        if args.use_h2o:
            logits = model.prefill_h2o(prompt_tensor, chunk_size=128)
        else:
            logits = model.prefill_chunked(prompt_tensor, chunk_size=128)

        # 2. Decode Loop
        for _ in range(args.max_new_tokens):
            next_logits = logits[0, -1, :].float()

            if args.temperature > 0:
                next_logits = next_logits / args.temperature

            if args.top_k > 0:
                topk_val, topk_idx = torch.topk(next_logits, min(args.top_k, next_logits.size(-1)))
                probs = torch.softmax(topk_val, dim=-1)
                chosen = torch.multinomial(probs, 1)
                next_token = topk_idx[chosen].item()
            else:
                next_token = torch.argmax(next_logits).item()

            generated.append(next_token)
            token_str = tok.decode([next_token])
            print(token_str, end="", flush=True)

            if next_token == tok.eos_token_id:
                break

            next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)

            if args.use_h2o:
                # Transactional H2O step
                for c in model.h2o_caches:
                    c.begin_block(1)

                logits, all_attn_probs = model.forward_block_h2o(next_input, return_attn_probs=True)

                for l_idx, c in enumerate(model.h2o_caches):
                    c.commit(1)
                    if all_attn_probs and l_idx < len(all_attn_probs):
                        c.update_scores(all_attn_probs[l_idx], 1)
                    c.evict_to_budget()
            else:
                logits, _ = model.forward_block(next_input)

    elapsed = time.time() - t0
    n_new = len(generated) - len(ids)
    tok_per_sec = n_new / max(elapsed, 1e-6)

    cache_str = "H2O (budget=%d tokens)" % budget if args.use_h2o else "Full KV"
    print("\n" + sep)
    print("[Stats] %d tokens | %.2fs | %.2f tok/s | Cache: %s | Device: %s" %
          (n_new, elapsed, tok_per_sec, cache_str, device.upper()))
    print(sep + "\n")
    return tok.decode(generated)


def main():
    args = parse_args()
    tok = load_tokenizer(os.path.expanduser(args.model_dir))
    model, device, dtype = load_model_instance(args)

    if args.prompt:
        generate_response(model, tok, args.prompt, args, device, dtype)
    else:
        print("\n" + "=" * 65)
        print("  EdgeSpec-H2O Chat Interface  (Type 'exit' to quit)")
        print("  Model: %s | Cache: H2O (%d%% budget) | Device: %s (%s)" %
              (model.config_name if hasattr(model, 'config_name') else "OPT",
               int(args.h2o_budget * 100), device.upper(), dtype))
        print("=" * 65)

        while True:
            try:
                user_input = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat. Goodbye!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting chat. Goodbye!")
                break
            generate_response(model, tok, user_input, args, device, dtype)


if __name__ == "__main__":
    main()
