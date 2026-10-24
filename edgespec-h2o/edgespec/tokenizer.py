# -*- coding: utf-8 -*-
"""Vendored GPT-2 BPE tokenizer for OPT.

Python-3.6 safe.  Uses OPT's vocab.json and merges.txt (GPT-2 BPE).
No dependency on ``transformers`` or ``tokenizers`` (Rust).

References
----------
- Original OpenAI encoder.py logic
- OPT uses the GPT-2 tokenizer with BOS = EOS = </s> (token id 2)
"""

from __future__ import unicode_literals

import json
import os
import re

try:
    import regex
    _REGEX = True
except ImportError:
    _REGEX = False


def _bytes_to_unicode():
    """Map bytes 0-255 to unicode strings (GPT-2 byte-level BPE)."""
    bs = (
        list(range(ord('!'), ord('~') + 1))
        + list(range(ord('\xa1'), ord('\xac') + 1))
        + list(range(ord('\xae'), ord('\xff') + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def _get_pairs(word):
    """Return set of symbol pairs in a word (tuple of symbols)."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# GPT-2 pre-tokenisation pattern
_GPT2_PAT_STR = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# Fallback for stdlib re (no \p{} support)
_GPT2_PAT_FALLBACK = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""


class GPT2BPETokenizer(object):
    """Minimal GPT-2 BPE tokenizer (CPU-only, vendored)."""

    def __init__(self, vocab_file, merges_file, bos_token_id=2, eos_token_id=2,
                 pad_token_id=1):
        """
        Parameters
        ----------
        vocab_file : str
            Path to vocab.json (GPT-2 style).
        merges_file : str
            Path to merges.txt (GPT-2 BPE merge rules).
        bos_token_id : int
            BOS token id (OPT uses </s> = 2).
        eos_token_id : int
            EOS token id.
        pad_token_id : int
            PAD token id.
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_file, 'r', encoding='utf-8') as f:
            bpe_data = f.read()
        merges_lines = bpe_data.split('\n')
        # Skip the header line (#version: ...)
        if merges_lines and merges_lines[0].startswith('#'):
            merges_lines = merges_lines[1:]
        merges = []
        for line in merges_lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    merges.append(tuple(parts))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.cache = {}

        if _REGEX:
            self.pat = regex.compile(_GPT2_PAT_STR)
        else:
            self.pat = re.compile(_GPT2_PAT_FALLBACK)

        self.vocab_size = len(self.encoder)

    def bpe(self, token):
        """Apply BPE to a single pre-tokenised word."""
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            return token

        while True:
            # Find the pair with the lowest rank
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text, add_bos=True):
        """Tokenize text to a list of integer token ids.

        Parameters
        ----------
        text : str
            Input text.
        add_bos : bool
            If True, prepend the BOS token (OPT convention).

        Returns
        -------
        list of int
        """
        bpe_tokens = []
        for match in self.pat.findall(text):
            # Encode each byte of the matched token to the byte-level BPE alphabet
            token = ''.join(self.byte_encoder[b] for b in match.encode('utf-8'))
            bpe_result = self.bpe(token)
            for bpe_token in bpe_result.split(' '):
                bpe_tokens.append(self.encoder[bpe_token])

        if add_bos:
            bpe_tokens = [self.bos_token_id] + bpe_tokens

        return bpe_tokens

    def decode(self, token_ids):
        """Decode a list of token ids to a string.

        Parameters
        ----------
        token_ids : list of int

        Returns
        -------
        str
        """
        # Filter special tokens
        text_tokens = []
        for tid in token_ids:
            if tid in (self.bos_token_id, self.eos_token_id, self.pad_token_id):
                continue
            text_tokens.append(tid)

        text = ''.join(self.decoder.get(t, '') for t in text_tokens)
        text_bytes = bytearray([self.byte_decoder[c] for c in text])
        text = text_bytes.decode('utf-8', errors='replace')
        return text

    def __call__(self, text, add_bos=True):
        """Convenience: tokenize text."""
        return self.encode(text, add_bos=add_bos)


def get_tokenizer(vocab_file=None, merges_file=None):
    """Factory: return a GPT2BPETokenizer with OPT's vocab/merges.

    Parameters
    ----------
    vocab_file : str or None
        Path to vocab.json.  Defaults to data/vocab.json.
    merges_file : str or None
        Path to merges.txt.  Defaults to data/merges.txt.

    Returns
    -------
    GPT2BPETokenizer
    """
    from edgespec.config import VOCAB_FILE, MERGES_FILE
    vocab_file = vocab_file or VOCAB_FILE
    merges_file = merges_file or MERGES_FILE
    return GPT2BPETokenizer(vocab_file, merges_file)
