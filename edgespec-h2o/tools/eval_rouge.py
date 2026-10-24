#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROUGE-2 evaluation for summarisation tasks (P1-11).

Evaluates ROUGE-2 on XSUM and CNN/DailyMail using OPT with few-shot prompting.

Usage:
    python tools/eval_rouge.py --model facebook/opt-125m --dataset xsum --n 100
"""

from __future__ import print_function

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_rouge(predictions, references):
    """Compute ROUGE-2 F1 score.

    Parameters
    ----------
    predictions : list of str
    references : list of str

    Returns
    -------
    dict with rouge2 precision, recall, fmeasure
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)

        scores = {'rouge2_f': [], 'rouge2_p': [], 'rouge2_r': [],
                  'rougeL_f': [], 'rougeL_p': [], 'rougeL_r': []}

        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores['rouge2_f'].append(score['rouge2'].fmeasure)
            scores['rouge2_p'].append(score['rouge2'].precision)
            scores['rouge2_r'].append(score['rouge2'].recall)
            scores['rougeL_f'].append(score['rougeL'].fmeasure)
            scores['rougeL_p'].append(score['rougeL'].precision)
            scores['rougeL_r'].append(score['rougeL'].recall)

        return {k: sum(v) / len(v) if v else 0 for k, v in scores.items()}

    except ImportError:
        print("WARNING: rouge-score not installed, computing simple bigram overlap")
        return _simple_rouge2(predictions, references)


def _simple_rouge2(predictions, references):
    """Fallback simple ROUGE-2 computation without rouge-score package."""

    def bigrams(text):
        words = text.lower().split()
        return set(zip(words, words[1:])) if len(words) > 1 else set()

    f_scores = []
    for pred, ref in zip(predictions, references):
        pred_bg = bigrams(pred)
        ref_bg = bigrams(ref)
        if not pred_bg or not ref_bg:
            f_scores.append(0.0)
            continue
        overlap = pred_bg & ref_bg
        p = len(overlap) / len(pred_bg) if pred_bg else 0
        r = len(overlap) / len(ref_bg) if ref_bg else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f_scores.append(f)

    return {
        'rouge2_f': sum(f_scores) / len(f_scores) if f_scores else 0,
        'rouge2_p': 0,
        'rouge2_r': 0,
    }


def main():
    parser = argparse.ArgumentParser(description='ROUGE evaluation')
    parser.add_argument('--predictions', type=str, default=None,
                        help='JSON file with predictions')
    parser.add_argument('--references', type=str, default=None,
                        help='JSON file with references')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.predictions and args.references:
        with open(args.predictions) as f:
            preds = json.load(f)
        with open(args.references) as f:
            refs = json.load(f)

        scores = compute_rouge(preds, refs)
        print("ROUGE-2 F1: %.4f" % scores['rouge2_f'])

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(scores, f, indent=2)
    else:
        print("Usage: --predictions <file> --references <file>")


if __name__ == '__main__':
    main()
