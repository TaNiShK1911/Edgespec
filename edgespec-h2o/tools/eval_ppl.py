#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WikiText-2 perplexity evaluation (P1-11).

Evaluates model perplexity on WikiText-2 test set using sliding-window approach.
Works on both laptop (with transformers) and Nano (with vendored model).

Usage:
    python tools/eval_ppl.py --model facebook/opt-125m --device cuda
    python tools/eval_ppl.py --weights data/weights/opt-1.3b-4bit.edgespec --device cuda
"""

from __future__ import print_function

import argparse
import json
import math
import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgespec.config import RESULTS_DIR


def eval_ppl_hf(model_name, device='cuda', n_windows=64, window_size=1024,
                stride=512, dtype=torch.float16):
    """Evaluate perplexity using HuggingFace model + dataset.

    Parameters
    ----------
    model_name : str
    device : str
    n_windows : int  (max number of windows to evaluate)
    window_size : int  (context window size)
    stride : int

    Returns
    -------
    dict with ppl and metadata
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    model.eval()

    # Load WikiText-2
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer.encode(text)

    print("WikiText-2 test: %d tokens" % len(encodings))

    nlls = []
    n_tokens = 0
    windows_done = 0

    for begin in range(0, len(encodings) - window_size, stride):
        if windows_done >= n_windows:
            break

        end = begin + window_size
        input_ids = torch.tensor([encodings[begin:end]], device=device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss.item()

        nlls.append(nll)
        n_tokens += window_size
        windows_done += 1

        if windows_done % 10 == 0:
            current_ppl = math.exp(sum(nlls) / len(nlls))
            print("  window %d/%d, running PPL=%.2f" % (
                windows_done, n_windows, current_ppl))

    ppl = math.exp(sum(nlls) / len(nlls))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "dataset": "wikitext-2-raw-v1",
        "ppl": ppl,
        "n_windows": windows_done,
        "window_size": window_size,
        "stride": stride,
        "dtype": str(dtype),
        "n_tokens_evaluated": n_tokens,
    }


def eval_ppl_vendored(model, tokenizer, device='cuda', n_windows=64,
                       window_size=1024, stride=512):
    """Evaluate perplexity using vendored OPT model.

    Parameters
    ----------
    model : OPTModel (our vendored model)
    tokenizer : GPT2BPETokenizer
    device : str
    n_windows : int
    window_size : int
    stride : int

    Returns
    -------
    dict with ppl
    """
    # Load WikiText-2 text
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n\n'.join(dataset['text'])
    except ImportError:
        # Fallback: look for local file
        wt2_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'wikitext2_test.txt'
        )
        with open(wt2_path, 'r') as f:
            text = f.read()

    encodings = tokenizer.encode(text, add_bos=False)
    print("WikiText-2 test: %d tokens" % len(encodings))

    nlls = []
    windows_done = 0

    for begin in range(0, len(encodings) - window_size, stride):
        if windows_done >= n_windows:
            break

        end = begin + window_size
        input_ids = torch.tensor(
            [encodings[begin:end]], device=device, dtype=torch.long
        )

        model.reset_caches()
        model.init_kv_caches(device=device)

        with torch.no_grad():
            logits = model.prefill_chunked(input_ids)
            # logits: [1, window_size, vocab_size]

            # Compute cross-entropy loss
            # Shift: predict token[i+1] from logits[i]
            shift_logits = logits[0, :-1, :].float()
            shift_labels = input_ids[0, 1:]

            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction='mean'
            )
            nlls.append(loss.item())

        windows_done += 1
        if windows_done % 10 == 0:
            current_ppl = math.exp(sum(nlls) / len(nlls))
            print("  window %d/%d, running PPL=%.2f" % (
                windows_done, n_windows, current_ppl))

    ppl = math.exp(sum(nlls) / len(nlls))

    return {
        "dataset": "wikitext-2-raw-v1",
        "ppl": ppl,
        "n_windows": windows_done,
        "window_size": window_size,
        "stride": stride,
    }


def main():
    parser = argparse.ArgumentParser(description='WikiText-2 PPL evaluation')
    parser.add_argument('--model', type=str, default=None,
                        help='HuggingFace model id (laptop mode)')
    parser.add_argument('--weights', type=str, default=None,
                        help='EdgeSpec weights file (Nano mode)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-windows', type=int, default=64)
    parser.add_argument('--window-size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'

    dtype = torch.float16 if args.dtype == 'float16' else torch.float32

    if args.model:
        result = eval_ppl_hf(
            args.model, args.device,
            n_windows=args.n_windows,
            window_size=args.window_size,
            stride=args.stride,
            dtype=dtype,
        )
    elif args.weights:
        from edgespec.model.loader import load_model
        from edgespec.tokenizer import get_tokenizer

        model = load_model(args.weights, device=args.device)
        tokenizer = get_tokenizer()
        result = eval_ppl_vendored(
            model, tokenizer, args.device,
            n_windows=args.n_windows,
            window_size=args.window_size,
            stride=args.stride,
        )
    else:
        print("Specify --model (HF) or --weights (EdgeSpec)")
        sys.exit(1)

    print("\n" + "=" * 40)
    print("PPL = %.2f" % result['ppl'])
    print("=" * 40)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print("Results saved to %s" % args.output)


if __name__ == '__main__':
    main()
