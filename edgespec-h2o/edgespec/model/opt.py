# -*- coding: utf-8 -*-
"""Vendored OPT forward pass — pure PyTorch, no ``transformers`` dependency.

Python-3.6 safe.  Supports k >= 1 new tokens per call (needed for spec-decode
verification in Phase 3).  Manual attention that optionally returns attention
probabilities (needed for H2O scoring in Phase 2).

Architecture (OPT-1.3B)
-----------------------
- Pre-LayerNorm (350M+), Post-LayerNorm (125M)
- Learned absolute positions with offset = 2
- ReLU FFN (no gated/SiLU)
- Tied LM head (embed_tokens weights reused)
- BOS = EOS = </s> (token id 2)
- MHA (32 heads for 1.3B, 12 for 125M)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """LayerNorm that supports Half on CPU by executing in float32 when necessary."""

    def forward(self, input):
        if self.weight is not None and input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        if input.is_cuda or input.dtype != torch.float16:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        w = self.weight.float() if self.weight is not None else None
        b = self.bias.float() if self.bias is not None else None
        return F.layer_norm(input.float(), self.normalized_shape, w, b, self.eps).half()



class KVCache(object):
    """Simple pre-allocated KV cache for one layer.

    Phase 1 uses full (non-evicting) cache.  Phase 2 replaces this with H2OCache.

    Parameters
    ----------
    n_heads : int
    d_head : int
    max_seq_len : int
    device : str
    dtype : torch.dtype
    """

    def __init__(self, n_heads, d_head, max_seq_len, device='cuda',
                 dtype=torch.float16):
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.length = 0

        # Preallocate: [n_heads, max_seq_len, d_head]
        self.k = torch.zeros(n_heads, max_seq_len, d_head, device=device, dtype=dtype)
        self.v = torch.zeros(n_heads, max_seq_len, d_head, device=device, dtype=dtype)

    def append(self, k_new, v_new):
        """Append new KV entries.

        Parameters
        ----------
        k_new : [n_heads, n_new, d_head]
        v_new : [n_heads, n_new, d_head]

        Returns
        -------
        k_full : [n_heads, length, d_head]  (view of the preallocated buffer)
        v_full : [n_heads, length, d_head]
        """
        n_new = k_new.shape[1]
        start = self.length
        end = self.length + n_new
        assert end <= self.max_seq_len, (
            "KV cache overflow: %d + %d > %d" % (self.length, n_new, self.max_seq_len)
        )

        self.k[:, start:end, :] = k_new
        self.v[:, start:end, :] = v_new
        self.length = end

        return self.k[:, :end, :], self.v[:, :end, :]

    def reset(self):
        """Clear the cache."""
        self.length = 0

    def memory_bytes(self):
        """Return bytes used by the K and V tensors."""
        elem_bytes = 2 if self.k.dtype == torch.float16 else 4
        return 2 * self.n_heads * self.max_seq_len * self.d_head * elem_bytes


class OPTAttention(nn.Module):
    """Multi-head attention for OPT.

    Manual implementation (no scaled_dot_product_attention — not in PyTorch 1.10).
    Optionally returns attention probabilities for H2O scoring.
    """

    def __init__(self, d_model, n_heads, dtype=torch.float16):
        super(OPTAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, kv_cache, positions, return_attn_probs=False):
        """
        Parameters
        ----------
        x : [1, n_new, d_model]  (batch=1 always)
        kv_cache : KVCache for this layer
        positions : [1, n_new] int  — absolute position ids
        return_attn_probs : bool

        Returns
        -------
        output : [1, n_new, d_model]
        attn_probs : [n_heads, n_new, seq_len] or None
        """
        bsz, n_new, _ = x.shape  # bsz = 1

        # Project Q, K, V for new tokens
        q = self.q_proj(x)  # [1, n_new, d_model]
        k_new = self.k_proj(x)
        v_new = self.v_proj(x)

        # Reshape to [n_heads, n_new, d_head]
        q = q.view(bsz, n_new, self.n_heads, self.d_head).permute(0, 2, 1, 3).squeeze(0)
        k_new = k_new.view(bsz, n_new, self.n_heads, self.d_head).permute(0, 2, 1, 3).squeeze(0)
        v_new = v_new.view(bsz, n_new, self.n_heads, self.d_head).permute(0, 2, 1, 3).squeeze(0)

        # Append to KV cache and get full K, V
        k_full, v_full = kv_cache.append(k_new, v_new)
        # k_full, v_full: [n_heads, seq_len, d_head]

        seq_len = k_full.shape[1]

        # Attention scores: [n_heads, n_new, seq_len]
        attn_weights = torch.matmul(q, k_full.transpose(1, 2)) * self.scale

        # Causal mask: each new token at position i can attend to positions 0..i
        # New tokens occupy positions [seq_len - n_new, ..., seq_len - 1]
        if n_new > 1:
            # Build causal mask for the new block
            # Token j (0-indexed in the new block) is at cache position (seq_len - n_new + j)
            # It can attend to cache positions 0..(seq_len - n_new + j)
            causal_mask = torch.ones(n_new, seq_len, device=attn_weights.device,
                                     dtype=torch.bool)
            for j in range(n_new):
                end_pos = seq_len - n_new + j + 1
                causal_mask[j, end_pos:] = False
            attn_weights = attn_weights.masked_fill(
                ~causal_mask.unsqueeze(0), float('-inf')
            )

        # Softmax in FP32 for numerical stability
        attn_probs = F.softmax(attn_weights.float(), dim=-1).to(x.dtype)

        # Weighted sum: [n_heads, n_new, d_head]
        attn_output = torch.matmul(attn_probs, v_full)

        # Reshape back: [1, n_new, d_model]
        attn_output = attn_output.unsqueeze(0).permute(0, 2, 1, 3).reshape(bsz, n_new, self.d_model)

        output = self.out_proj(attn_output)

        if return_attn_probs:
            return output, attn_probs
        return output, None


class OPTMLP(nn.Module):
    """OPT feed-forward network: Linear → ReLU → Linear."""

    def __init__(self, d_model, d_ffn, dtype=torch.float16):
        super(OPTMLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ffn, bias=True)
        self.fc2 = nn.Linear(d_ffn, d_model, bias=True)

    def forward(self, x):
        h = self.fc1(x)
        if not h.is_cuda and h.dtype == torch.float16:
            h = F.relu(h.float()).half()
        else:
            h = F.relu(h)
        return self.fc2(h)


class OPTDecoderLayer(nn.Module):
    """One transformer block for OPT (pre-LN variant for 350M+)."""

    def __init__(self, d_model, n_heads, d_ffn, pre_layernorm=True,
                 layer_norm_eps=1e-5, dtype=torch.float16):
        super(OPTDecoderLayer, self).__init__()
        self.pre_layernorm = pre_layernorm

        self.self_attn_layer_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = OPTAttention(d_model, n_heads, dtype=dtype)

        self.final_layer_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = OPTMLP(d_model, d_ffn, dtype=dtype)

    def __setattr__(self, name, value):
        if name in ("fc1", "fc2") and hasattr(self, "mlp"):
            setattr(self.mlp, name, value)
            return
        super(OPTDecoderLayer, self).__setattr__(name, value)

    @property
    def fc1(self):
        return self.mlp.fc1

    @fc1.setter
    def fc1(self, mod):
        self.mlp.fc1 = mod

    @property
    def fc2(self):
        return self.mlp.fc2

    @fc2.setter
    def fc2(self, mod):
        self.mlp.fc2 = mod

    def forward(self, x, kv_cache, positions, return_attn_probs=False):
        """
        Parameters
        ----------
        x : [1, n_new, d_model]
        kv_cache : KVCache
        positions : [1, n_new]
        return_attn_probs : bool

        Returns
        -------
        x : [1, n_new, d_model]
        attn_probs : [n_heads, n_new, seq_len] or None
        """
        residual = x

        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        x, attn_probs = self.self_attn(x, kv_cache, positions,
                                        return_attn_probs=return_attn_probs)
        x = residual + x

        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)

        x = self.mlp(x)
        x = residual + x

        if not self.pre_layernorm:
            x = self.final_layer_norm(x)

        return x, attn_probs


class OPTModel(nn.Module):
    """Full vendored OPT causal language model.

    No dependency on ``transformers``.  Supports:
    - k >= 1 new tokens per forward (for spec-decode verification)
    - Layer-major chunked prefill
    - Optional attention probability return (for H2O)
    """

    def __init__(self, config, dtype=torch.float16):
        """
        Parameters
        ----------
        config : dict
            One of the dicts from edgespec.config.OPT_CONFIGS.
        dtype : torch.dtype
        """
        super(OPTModel, self).__init__()
        self.config = config
        self.dtype = dtype

        d_model = config['d_model']
        n_heads = config['n_heads']
        n_layers = config['n_layers']
        d_ffn = config['d_ffn']
        vocab_size = config['vocab_size']
        max_positions = config['max_positions']
        pre_ln = config.get('pre_layernorm', True)
        ln_eps = config.get('layer_norm_eps', 1e-5)

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_layers = n_layers
        self.max_positions = max_positions
        self.position_offset = config.get('position_offset', 2)

        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embed_positions = nn.Embedding(max_positions + self.position_offset,
                                            d_model)

        # Optional project_in/project_out (only for some OPT variants)
        # OPT-125M, 1.3B don't use these — d_model == word_embed_proj_dim

        # Decoder layers
        self.layers = nn.ModuleList([
            OPTDecoderLayer(d_model, n_heads, d_ffn, pre_layernorm=pre_ln,
                            layer_norm_eps=ln_eps, dtype=dtype)
            for _ in range(n_layers)
        ])

        # Final layer norm (pre-LN models have this before the LM head)
        if pre_ln:
            self.final_layer_norm = LayerNorm(d_model, eps=ln_eps)
        else:
            self.final_layer_norm = None

        # LM head — tied to embed_tokens (set up after loading)
        self.lm_head_weight = None  # will point to embed_tokens.weight

        # KV caches (one per layer)
        self.kv_caches = None

        if dtype is not None and dtype != torch.float32:
            self.to(dtype)

    def init_kv_caches(self, max_seq_len=None, device='cuda'):
        """Pre-allocate KV caches for all layers.

        Parameters
        ----------
        max_seq_len : int or None
            Defaults to self.max_positions.
        device : str
        """
        if max_seq_len is None:
            max_seq_len = self.max_positions
        self.kv_caches = [
            KVCache(self.n_heads, self.d_head, max_seq_len, device=device,
                    dtype=self.dtype)
            for _ in range(self.n_layers)
        ]

    def reset_caches(self):
        """Clear all KV caches."""
        if self.kv_caches is not None:
            for cache in self.kv_caches:
                cache.reset()
        for m in self.modules():
            if hasattr(m, '_cached_weight'):
                m._cached_weight = None

    def forward_block(self, token_ids, start_pos=None, return_attn_probs=False):
        """Forward pass for k >= 1 new tokens.

        Parameters
        ----------
        token_ids : torch.LongTensor [1, k]
            Token ids for the new tokens.
        start_pos : int or None
            Starting position for these tokens.  If None, uses current cache length.
        return_attn_probs : bool
            If True, returns per-layer attention probs (for H2O).

        Returns
        -------
        logits : [1, k, vocab_size]
        all_attn_probs : list of [n_heads, k, seq_len] or None
        """
        assert self.kv_caches is not None, "Call init_kv_caches() first"

        device = token_ids.device
        k = token_ids.shape[1]

        if start_pos is None:
            start_pos = self.kv_caches[0].length

        # Position ids with OPT offset
        positions = torch.arange(
            start_pos + self.position_offset,
            start_pos + self.position_offset + k,
            device=device, dtype=torch.long
        ).unsqueeze(0)  # [1, k]

        # Token + position embeddings
        x = self.embed_tokens(token_ids) + self.embed_positions(positions)
        x = x.to(self.dtype)

        all_attn_probs = [] if return_attn_probs else None

        for i, layer in enumerate(self.layers):
            x, attn_probs = layer(x, self.kv_caches[i], positions,
                                  return_attn_probs=return_attn_probs)
            if return_attn_probs:
                all_attn_probs.append(attn_probs)

        # Final layer norm
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        # LM head (tied weights)
        if self.lm_head_weight is not None:
            logits = F.linear(x.to(self.lm_head_weight.dtype), self.lm_head_weight)
        else:
            logits = F.linear(x.to(self.embed_tokens.weight.dtype),
                              self.embed_tokens.weight)

        return logits, all_attn_probs

    def prefill_chunked(self, token_ids, chunk_size=128, return_attn_probs=False):
        """Layer-major chunked prefill.

        Processes all prompt tokens through layer l, then layer l+1, etc.
        Within each layer, attention is computed in chunks of ``chunk_size``
        tokens to limit peak memory (score matrix is chunk_size × seq_len).

        Parameters
        ----------
        token_ids : torch.LongTensor [1, seq_len]
        chunk_size : int
        return_attn_probs : bool

        Returns
        -------
        logits : [1, seq_len, vocab_size]  (only last position is useful)
        """
        assert self.kv_caches is not None, "Call init_kv_caches() first"
        self.reset_caches()

        device = token_ids.device
        seq_len = token_ids.shape[1]

        # Position ids
        positions = torch.arange(
            self.position_offset,
            self.position_offset + seq_len,
            device=device, dtype=torch.long
        ).unsqueeze(0)  # [1, seq_len]

        # Token + position embeddings
        x = self.embed_tokens(token_ids) + self.embed_positions(positions)
        x = x.to(self.dtype)

        # Layer-major: process all tokens through each layer sequentially
        for layer_idx, layer in enumerate(self.layers):
            cache = self.kv_caches[layer_idx]
            cache.reset()  # ensure clean

            # Process in chunks
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk = x[:, start:end, :]  # [1, chunk_len, d_model]
                chunk_positions = positions[:, start:end]

                chunk_out, _ = layer(chunk, cache, chunk_positions,
                                     return_attn_probs=False)
                x[:, start:end, :] = chunk_out

        # Final layer norm
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        # LM head
        if self.lm_head_weight is not None:
            logits = F.linear(x.to(self.lm_head_weight.dtype), self.lm_head_weight)
        else:
            logits = F.linear(x.to(self.embed_tokens.weight.dtype),
                              self.embed_tokens.weight)

        return logits

    @torch.no_grad()
    def generate_greedy(self, prompt_ids, max_new_tokens=128, chunk_size=128):
        """Simple greedy generation (Phase 1 baseline).

        Parameters
        ----------
        prompt_ids : torch.LongTensor [1, prompt_len]
        max_new_tokens : int
        chunk_size : int  (for prefill)

        Returns
        -------
        generated_ids : list of int  (prompt + generated)
        """
        device = prompt_ids.device

        # Prefill
        logits = self.prefill_chunked(prompt_ids, chunk_size=chunk_size)
        # logits: [1, prompt_len, vocab_size]
        next_token = logits[0, -1, :].argmax(dim=-1).item()

        all_ids = prompt_ids[0].tolist() + [next_token]

        # Decode
        for _ in range(max_new_tokens - 1):
            if next_token == self.config.get('eos_token_id', 2):
                break

            input_ids = torch.tensor([[next_token]], device=device, dtype=torch.long)
            logits, _ = self.forward_block(input_ids)
            next_token = logits[0, -1, :].argmax(dim=-1).item()
            all_ids.append(next_token)

        return all_ids

    def total_kv_memory_bytes(self):
        """Total bytes used by all KV caches."""
        if self.kv_caches is None:
            return 0
        return sum(c.memory_bytes() for c in self.kv_caches)

    # =======================================================================
    #  H2O cache integration (Phase 2)
    # =======================================================================

    def init_h2o_caches(self, n_hh=128, n_local=128, n_pin=4, n_scratch=8,
                        decay=0.98, evict_every=1, device='cuda'):
        """Pre-allocate H2O caches for all layers.

        Parameters
        ----------
        n_hh, n_local, n_pin, n_scratch, decay, evict_every : see H2OCache
        device : str
        """
        from edgespec.cache.h2o_cache import create_h2o_caches

        self.h2o_caches = create_h2o_caches(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_head=self.d_head,
            n_hh=n_hh, n_local=n_local,
            n_pin=n_pin, n_scratch=n_scratch,
            decay=decay, evict_every=evict_every,
            device=device, dtype=self.dtype,
        )

    def reset_h2o_caches(self):
        """Clear all H2O caches."""
        if hasattr(self, 'h2o_caches') and self.h2o_caches is not None:
            for cache in self.h2o_caches:
                cache.reset()

    def _h2o_attention(self, layer_idx, x, h2o_cache, positions,
                       return_attn_probs=True):
        """Run attention with H2O cache for one layer.

        Instead of using the layer's built-in KV cache, we use the H2O cache
        which handles append_kv, masking, and provides valid-slot semantics.

        Parameters
        ----------
        layer_idx : int
        x : [1, n_new, d_model]
        h2o_cache : H2OCache
        positions : [n_new] int tensor — absolute position ids for new tokens
        return_attn_probs : bool

        Returns
        -------
        output : [1, n_new, d_model]
        attn_probs : [n_heads, n_new, n_total] or None
        """
        layer = self.layers[layer_idx]
        attn = layer.self_attn
        bsz = 1
        n_new = x.shape[1]

        # Project Q, K, V
        q = attn.q_proj(x)
        k_new = attn.k_proj(x)
        v_new = attn.v_proj(x)

        # Reshape to [n_heads, n_new, d_head]
        q = q.view(bsz, n_new, attn.n_heads, attn.d_head).permute(0, 2, 1, 3).squeeze(0)
        k_new = k_new.view(bsz, n_new, attn.n_heads, attn.d_head).permute(0, 2, 1, 3).squeeze(0)
        v_new = v_new.view(bsz, n_new, attn.n_heads, attn.d_head).permute(0, 2, 1, 3).squeeze(0)

        # Append new KV to H2O cache
        h2o_cache.append_kv(k_new, v_new, positions)

        # Get all valid KV from cache
        k_full, v_full, valid_mask, n_total = h2o_cache.get_kv()

        # Attention scores: [n_heads, n_new, n_total]
        attn_weights = torch.matmul(q, k_full.transpose(1, 2)) * attn.scale

        # Valid-slot mask: only attend to valid slots
        # valid_mask: [n_heads, n_total] -> [n_heads, 1, n_total]
        attn_weights = attn_weights.masked_fill(
            ~valid_mask.unsqueeze(1), float('-inf')
        )

        # Causal mask: each new token at position i can only attend to
        # positions <= its own position
        if n_new > 1:
            cache_positions = h2o_cache.get_positions()  # [n_heads, n_total]
            new_positions = positions.unsqueeze(0).expand(self.n_heads, -1)  # [n_heads, n_new]
            # For each head h, new token j at position new_positions[h,j]
            # can attend to cache position cache_positions[h,k] if
            # cache_positions[h,k] <= new_positions[h,j]
            causal_mask = cache_positions.unsqueeze(1) <= new_positions.unsqueeze(2)
            # causal_mask: [n_heads, n_new, n_total]
            attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))

        # Softmax in FP32
        attn_probs = F.softmax(attn_weights.float(), dim=-1).to(x.dtype)

        # Weighted sum
        attn_output = torch.matmul(attn_probs, v_full)

        # Reshape back
        attn_output = attn_output.unsqueeze(0).permute(0, 2, 1, 3).reshape(
            bsz, n_new, attn.d_model
        )
        output = attn.out_proj(attn_output)

        if return_attn_probs:
            return output, attn_probs
        return output, None

    def forward_block_h2o(self, token_ids, start_pos=None,
                           return_attn_probs=True):
        """Forward pass using H2O caches.

        Does NOT call begin_block/commit — caller must do that for
        speculative rollback safety.

        Parameters
        ----------
        token_ids : torch.LongTensor [1, k]
        start_pos : int or None
        return_attn_probs : bool

        Returns
        -------
        logits : [1, k, vocab_size]
        all_attn_probs : list of [n_heads, k, n_total] or None
        """
        assert hasattr(self, 'h2o_caches') and self.h2o_caches is not None, \
            "Call init_h2o_caches() first"

        device = token_ids.device
        k = token_ids.shape[1]

        if start_pos is None:
            start_pos = self.h2o_caches[0].length

        # Position ids with OPT offset
        positions = torch.arange(
            start_pos + self.position_offset,
            start_pos + self.position_offset + k,
            device=device, dtype=torch.long
        )

        # Token + position embeddings
        position_ids = positions.unsqueeze(0)  # [1, k]
        x = self.embed_tokens(token_ids) + self.embed_positions(position_ids)
        x = x.to(self.dtype)

        all_attn_probs = [] if return_attn_probs else None

        for i, layer in enumerate(self.layers):
            residual = x
            if layer.pre_layernorm:
                x = layer.self_attn_layer_norm(x)

            x, attn_probs = self._h2o_attention(
                i, x, self.h2o_caches[i], positions,
                return_attn_probs=return_attn_probs
            )

            x = residual + x

            if not layer.pre_layernorm:
                x = layer.self_attn_layer_norm(x)

            residual = x
            if layer.pre_layernorm:
                x = layer.final_layer_norm(x)

            x = layer.mlp(x)
            x = residual + x

            if not layer.pre_layernorm:
                x = layer.final_layer_norm(x)

            if return_attn_probs:
                all_attn_probs.append(attn_probs)

        # Final layer norm
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        # LM head
        if self.lm_head_weight is not None:
            logits = F.linear(x.to(self.lm_head_weight.dtype), self.lm_head_weight)
        else:
            logits = F.linear(x.to(self.embed_tokens.weight.dtype),
                              self.embed_tokens.weight)

        return logits, all_attn_probs

    def prefill_h2o(self, token_ids, chunk_size=128):
        """Layer-major chunked prefill with H2O on-the-fly compression.

        For each layer:
        1. Run full-attention prefill (all tokens)
        2. Accumulate attention scores across chunks
        3. Compress KV to budget using H2O eviction

        Parameters
        ----------
        token_ids : torch.LongTensor [1, seq_len]
        chunk_size : int

        Returns
        -------
        logits : [1, seq_len, vocab_size]
        """
        assert hasattr(self, 'h2o_caches') and self.h2o_caches is not None, \
            "Call init_h2o_caches() first"
        self.reset_h2o_caches()

        device = token_ids.device
        seq_len = token_ids.shape[1]

        # Position ids
        positions = torch.arange(
            self.position_offset,
            self.position_offset + seq_len,
            device=device, dtype=torch.long
        )
        position_ids = positions.unsqueeze(0)  # [1, seq_len]

        # Token + position embeddings
        x = self.embed_tokens(token_ids) + self.embed_positions(position_ids)
        x = x.to(self.dtype)

        for layer_idx, layer in enumerate(self.layers):
            cache = self.h2o_caches[layer_idx]
            cache.reset()

            attn = layer.self_attn
            bsz = 1

            # Pre-LN
            residual = x
            if layer.pre_layernorm:
                h = layer.self_attn_layer_norm(x)
            else:
                h = x

            # Project full Q, K, V
            q_full = attn.q_proj(h)
            k_full_proj = attn.k_proj(h)
            v_full_proj = attn.v_proj(h)

            # Reshape
            q_full = q_full.view(bsz, seq_len, attn.n_heads, attn.d_head
                                 ).permute(0, 2, 1, 3).squeeze(0)
            k_full_proj = k_full_proj.view(bsz, seq_len, attn.n_heads, attn.d_head
                                           ).permute(0, 2, 1, 3).squeeze(0)
            v_full_proj = v_full_proj.view(bsz, seq_len, attn.n_heads, attn.d_head
                                           ).permute(0, 2, 1, 3).squeeze(0)

            # Full causal attention in chunks to limit memory
            attn_output = torch.zeros(self.n_heads, seq_len, attn.d_head,
                                      device=device, dtype=self.dtype)
            score_accum = torch.zeros(self.n_heads, seq_len,
                                      device=device, dtype=torch.float32)

            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                q_chunk = q_full[:, start:end, :]  # [n_heads, chunk, d_head]

                # Attention with all K up to position 'end'
                # Causal: token at position i attends to 0..i
                k_slice = k_full_proj[:, :end, :]  # [n_heads, end, d_head]
                v_slice = v_full_proj[:, :end, :]

                scores = torch.matmul(q_chunk, k_slice.transpose(1, 2)) * attn.scale
                # [n_heads, chunk, end]

                # Causal mask
                chunk_len = end - start
                causal = torch.ones(chunk_len, end, device=device, dtype=torch.bool)
                for j in range(chunk_len):
                    causal[j, start + j + 1:] = False
                scores = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))

                probs = F.softmax(scores.float(), dim=-1).to(self.dtype)
                attn_output[:, start:end, :] = torch.matmul(probs, v_slice)

                # Accumulate scores: sum probs across query positions
                # For initial H2O scoring, we sum how much each key position
                # was attended to across all queries in this chunk
                score_accum[:, :end] += probs.float().sum(dim=1)

            # Output projection
            attn_out = attn_output.unsqueeze(0).permute(0, 2, 1, 3).reshape(
                bsz, seq_len, attn.d_model
            )
            attn_out = attn.out_proj(attn_out)

            x = residual + attn_out
            if not layer.pre_layernorm:
                x = layer.self_attn_layer_norm(x)

            # FFN
            residual = x
            if layer.pre_layernorm:
                x = layer.final_layer_norm(x)
            x = layer.mlp(x)
            x = residual + x
            if not layer.pre_layernorm:
                x = layer.final_layer_norm(x)

            # Initialize H2O cache with prefill KV and compress to budget
            cache.init_from_prefill(k_full_proj, v_full_proj, positions, score_accum)

        # Final layer norm
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        # LM head
        if self.lm_head_weight is not None:
            logits = F.linear(x.to(self.lm_head_weight.dtype), self.lm_head_weight)
        else:
            logits = F.linear(x.to(self.embed_tokens.weight.dtype),
                              self.embed_tokens.weight)

        return logits

    @torch.no_grad()
    def generate_greedy_h2o(self, prompt_ids, max_new_tokens=128,
                             chunk_size=128):
        """Greedy generation using H2O cache (Phase 2 baseline).

        Parameters
        ----------
        prompt_ids : torch.LongTensor [1, prompt_len]
        max_new_tokens : int
        chunk_size : int

        Returns
        -------
        generated_ids : list of int (prompt + generated)
        """
        device = prompt_ids.device

        # Prefill with H2O
        logits = self.prefill_h2o(prompt_ids, chunk_size=chunk_size)
        next_token = logits[0, -1, :].argmax(dim=-1).item()

        all_ids = prompt_ids[0].tolist() + [next_token]

        for _ in range(max_new_tokens - 1):
            if next_token == self.config.get('eos_token_id', 2):
                break

            input_ids = torch.tensor([[next_token]], device=device,
                                     dtype=torch.long)

            # Begin block for 1 token
            for cache in self.h2o_caches:
                cache.begin_block(1)

            # Forward
            logits, all_probs = self.forward_block_h2o(input_ids)
            next_token = logits[0, -1, :].argmax(dim=-1).item()
            all_ids.append(next_token)

            # Commit 1 token, update scores, evict
            for i, cache in enumerate(self.h2o_caches):
                cache.commit(1)
                if all_probs is not None and all_probs[i] is not None:
                    cache.update_scores(all_probs[i], 1)
                cache.evict_to_budget()

        return all_ids

    def total_h2o_memory_bytes(self):
        """Total bytes used by all H2O caches."""
        if not hasattr(self, 'h2o_caches') or self.h2o_caches is None:
            return 0
        return sum(c.memory_bytes() for c in self.h2o_caches)

