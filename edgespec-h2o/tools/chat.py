#!/usr/bin/env python3
"""
Interactive / single-shot OPT inference on Jetson Nano (CPU, float32).

Usage
-----
# single prompt
python tools/chat.py --model-dir ~/edgespec/models/opt-125m \
                     --prompt "Once upon a time"

# interactive REPL  (type 'exit' to quit)
python tools/chat.py --model-dir ~/edgespec/models/opt-125m
"""
import argparse
import os
import sys
import time
import json
import struct

# ------------------------------------------------------------------ helpers --

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",
                   default=os.path.expanduser("~/edgespec/models/opt-125m"))
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--prompt", default=None,
                   help="Single prompt (non-interactive)")
    return p.parse_args()


# ---------------------------------------------------------------- model load --

def load_everything(model_dir):
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from edgespec.config import OPT_CONFIGS
    from edgespec.model.opt import OPTModel
    from edgespec.tokenizer import get_tokenizer
    from edgespec.model.loader import _assign_tensor

    vocab_file  = os.path.join(model_dir, "vocab.json")
    merges_file = os.path.join(model_dir, "merges.txt")
    print("[+] Loading tokenizer ...")
    tok = get_tokenizer(vocab_file=vocab_file, merges_file=merges_file)

    # detect config from config.json in model dir
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        hf_cfg = json.load(f)

    n_layers = hf_cfg.get("num_hidden_layers", 12)
    config_name = "opt-125m"
    for k, v in OPT_CONFIGS.items():
        if v.get("n_layers") == n_layers:
            config_name = k
            break

    config = OPT_CONFIGS[config_name]
    print("[+] Config: %s  (%d layers)" % (config_name, config["n_layers"]))

    print("[+] Building model skeleton ...")
    dtype = torch.float32
    model = OPTModel(config, dtype=dtype)

    # Load pytorch_model.bin  (HF state_dict format)
    ckpt_path = os.path.join(model_dir, "pytorch_model.bin")
    print("[+] Loading weights from %s  (%.0f MiB) ..." % (
        ckpt_path, os.path.getsize(ckpt_path) / 1024 / 1024))
    t0 = time.time()
    sd = torch.load(ckpt_path, map_location="cpu")
    print("[+] Checkpoint loaded in %.1fs" % (time.time() - t0))

    # Map HF keys -> our model
    mapping = {}
    mapping["embed_tokens.weight"] = sd.get("model.decoder.embed_tokens.weight")
    mapping["embed_positions.weight"] = sd.get("model.decoder.embed_positions.weight")
    if "model.decoder.final_layer_norm.weight" in sd:
        mapping["final_layer_norm.weight"] = sd["model.decoder.final_layer_norm.weight"]
        mapping["final_layer_norm.bias"]   = sd["model.decoder.final_layer_norm.bias"]

    for i in range(config["n_layers"]):
        hf_pfx  = "model.decoder.layers.%d" % i
        our_pfx = "layers.%d" % i
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            mapping["%s.self_attn.%s.weight" % (our_pfx, proj)] = \
                sd.get("%s.self_attn.%s.weight" % (hf_pfx, proj))
            mapping["%s.self_attn.%s.bias" % (our_pfx, proj)] = \
                sd.get("%s.self_attn.%s.bias" % (hf_pfx, proj))
        for ln in ("self_attn_layer_norm", "final_layer_norm"):
            mapping["%s.%s.weight" % (our_pfx, ln)] = \
                sd.get("%s.%s.weight" % (hf_pfx, ln))
            mapping["%s.%s.bias" % (our_pfx, ln)] = \
                sd.get("%s.%s.bias" % (hf_pfx, ln))
        mapping["%s.mlp.fc1.weight" % our_pfx] = sd.get("%s.fc1.weight" % hf_pfx)
        mapping["%s.mlp.fc1.bias"   % our_pfx] = sd.get("%s.fc1.bias"   % hf_pfx)
        mapping["%s.mlp.fc2.weight" % our_pfx] = sd.get("%s.fc2.weight" % hf_pfx)
        mapping["%s.mlp.fc2.bias"   % our_pfx] = sd.get("%s.fc2.bias"   % hf_pfx)

    print("[+] Assigning weights ...")
    for name, tensor in mapping.items():
        if tensor is not None:
            _assign_tensor(model, name, tensor.to(dtype))

    # Tie LM head to embedding
    model.lm_head_weight = model.embed_tokens.weight
    model.eval()
    print("[+] Model ready!\n")
    return model, tok


# --------------------------------------------------------------- generation --

def generate(model, tok, prompt_text, max_new_tokens, temperature, top_k):
    import torch

    ids = tok.encode(prompt_text)

    sep = "=" * 62
    print("\n" + sep)
    print("Prompt  : %s" % prompt_text)
    print(sep)
    print("Output  : ", end="", flush=True)

    generated = list(ids)
    t0 = time.time()

    # Reset and re-init KV caches for each generation
    model.init_kv_caches(max_seq_len=512, device="cpu")

    with torch.no_grad():
        # Prefill the prompt
        prompt_tensor = torch.tensor([ids], dtype=torch.long)
        logits = model.prefill_chunked(prompt_tensor, chunk_size=128)
        # logits: (1, prompt_len, vocab)

        for _ in range(max_new_tokens):
            next_logits = logits[0, -1, :].float()

            if temperature > 0:
                next_logits = next_logits / temperature

            if top_k > 0:
                topk_val, topk_idx = torch.topk(
                    next_logits, min(top_k, next_logits.size(-1)))
                probs = torch.softmax(topk_val, dim=-1)
                chosen = torch.multinomial(probs, 1)
                next_token = topk_idx[chosen].item()
            else:
                next_token = torch.argmax(next_logits).item()

            generated.append(next_token)
            print(tok.decode([next_token]), end="", flush=True)

            if next_token == tok.eos_token_id:
                break

            # Decode step: one token at a time
            next_input = torch.tensor([[next_token]], dtype=torch.long)
            logits, _ = model.forward_block(next_input)

    elapsed = time.time() - t0
    n_new = len(generated) - len(ids)
    print("\n" + sep)
    print("[stats] %d new tokens | %.2fs | %.1f tok/s" % (
        n_new, elapsed, n_new / max(elapsed, 1e-6)))
    print(sep + "\n")


# ----------------------------------------------------------------------- main

def main():
    args = parse_args()
    model, tok = load_everything(args.model_dir)

    if args.prompt:
        generate(model, tok, args.prompt,
                 args.max_new_tokens, args.temperature, args.top_k)
    else:
        print("=" * 62)
        print("  OPT Interactive Chat  |  type 'exit' to quit")
        print("=" * 62)
        while True:
            try:
                user_input = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Bye!")
                break
            generate(model, tok, user_input,
                     args.max_new_tokens, args.temperature, args.top_k)


if __name__ == "__main__":
    main()
