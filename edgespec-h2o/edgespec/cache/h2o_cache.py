# -*- coding: utf-8 -*-
"""Time-decayed H2O KV cache with rollback-safe commit API.

Python-3.6 safe.  Implements the H2O (Heavy-Hitter Oracle) KV cache from
Zhang et al. (arXiv 2306.14048) with the following extensions:

1. **Per-head eviction** (Appendix A circular-queue)
2. **Time-decay** on accumulated attention scores (score <- lambda * score + attn)
3. **Pin-first-N** tokens to protect attention sinks
4. **Scratch slots** for speculative decoding commit/rollback (Phase 3)
5. **Token-importance vector** for the speculative drafter (Phase 3)

Layout per layer
----------------
K, V : [n_heads, capacity, d_head]  FP16
pos  : [n_heads, capacity]          int32  (absolute position ids)
score: [n_heads, capacity]          FP32   (accumulated attention scores)
valid: [n_heads, capacity]          bool   (slot in use)

capacity = n_pin + n_hh + n_local + n_scratch
"""

import math
import torch


class H2OCache(object):
    """H2O KV cache for one transformer layer.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    d_head : int
        Dimension per head.
    n_hh : int
        Number of heavy-hitter slots per head.
    n_local : int
        Number of local (recent) window slots per head.
    n_pin : int
        Number of pinned (attention-sink) slots at the start.
    n_scratch : int
        Number of scratch slots for speculative tokens (gamma_max + 1).
    decay : float
        Score decay factor lambda (1.0 = plain H2O, <1 = time-decayed).
    evict_every : int
        Evict only every N committed tokens to amortise cost.
    device : str
    dtype : torch.dtype
    """

    def __init__(self, n_heads, d_head, n_hh=128, n_local=128, n_pin=4,
                 n_scratch=8, decay=0.98, evict_every=1,
                 device='cuda', dtype=torch.float16):
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_hh = n_hh
        self.n_local = n_local
        self.n_pin = n_pin
        self.n_scratch = n_scratch
        self.decay = decay
        self.evict_every = evict_every
        self.device = device
        self.dtype = dtype

        # Budget = total tokens we keep (excluding scratch)
        self.budget = n_pin + n_hh + n_local

        # Capacity = budget + scratch
        self.capacity = self.budget + n_scratch

        # Storage -----------------------------------------------------------
        # K, V: [n_heads, capacity, d_head]
        self.k = torch.zeros(n_heads, self.capacity, d_head,
                             device=device, dtype=dtype)
        self.v = torch.zeros(n_heads, self.capacity, d_head,
                             device=device, dtype=dtype)

        # Position ids (absolute): [n_heads, capacity] — int32
        self.pos = torch.full((n_heads, self.capacity), -1,
                              device=device, dtype=torch.int32)

        # Accumulated attention scores: [n_heads, capacity] — FP32
        self.score = torch.zeros(n_heads, self.capacity,
                                 device=device, dtype=torch.float32)

        # Valid mask: [n_heads, capacity]
        self.valid = torch.zeros(n_heads, self.capacity,
                                 device=device, dtype=torch.bool)

        # Per-head slot counts (how many valid slots each head uses)
        # All heads share the same logical length initially
        self.length = 0  # total committed tokens seen so far

        # Scratch tracking
        self._scratch_start = 0  # where scratch slots begin in the buffer
        self._scratch_count = 0  # how many scratch slots are in use
        self._pending_commit = False

        # Eviction counter (for evict_every)
        self._commits_since_evict = 0

        # Token importance for the drafter (Phase 3): [max_positions]
        # Accumulated from attention probs, decayed
        self._importance = None  # lazily allocated

    # -----------------------------------------------------------------------
    #  Buffer management helpers
    # -----------------------------------------------------------------------

    def _n_valid(self):
        """Number of valid (non-scratch) slots per head. Uses head 0 as representative."""
        return int(self.valid[0].sum().item()) - self._scratch_count

    def _find_free_slots(self, n):
        """Find n contiguous free slots in the buffer. Returns start index."""
        # Simple strategy: append after last valid slot
        # Because we maintain a compact buffer, the first invalid slot after
        # all valid ones is our insertion point.
        n_used = self._n_valid() + self._scratch_count
        if n_used + n > self.capacity:
            raise RuntimeError(
                "H2OCache overflow: %d used + %d new > %d capacity. "
                "Run evict_to_budget() first." % (n_used, n, self.capacity)
            )
        return n_used

    # -----------------------------------------------------------------------
    #  Core API (matches plan §2.2)
    # -----------------------------------------------------------------------

    def begin_block(self, n_new):
        """Reserve scratch slots for n_new speculative tokens.

        Must be called before forward_block.  After forward, call commit().

        Parameters
        ----------
        n_new : int
            Number of new tokens (gamma + 1 for speculative, or 1 for normal).
        """
        if self._pending_commit:
            raise RuntimeError("begin_block called with pending commit — "
                               "call commit() or rollback() first")

        if n_new > self.n_scratch:
            raise ValueError(
                "n_new=%d > n_scratch=%d — increase n_scratch in config" %
                (n_new, self.n_scratch)
            )

        n_valid = self._n_valid()
        if n_valid + n_new > self.capacity:
            raise RuntimeError(
                "H2OCache overflow: %d valid + %d new > %d capacity. "
                "Run evict_to_budget() first." % (n_valid, n_new, self.capacity)
            )

        self._scratch_start = n_valid
        self._scratch_count = n_new
        self._pending_commit = True

        return self._scratch_start

    def append_kv(self, k_new, v_new, positions):
        """Write K, V for new tokens into scratch slots.

        Called by the model's attention layer during forward_block.

        Parameters
        ----------
        k_new : [n_heads, n_new, d_head]
        v_new : [n_heads, n_new, d_head]
        positions : [n_new]  int tensor — absolute position ids
        """
        n_new = k_new.shape[1]
        start = self._scratch_start
        end = start + n_new

        self.k[:, start:end, :] = k_new
        self.v[:, start:end, :] = v_new

        # Positions are the same across all heads
        for h in range(self.n_heads):
            self.pos[h, start:end] = positions.to(torch.int32)

        self.valid[:, start:end] = True

    def get_kv(self):
        """Return valid K, V, positions, and a validity mask for attention.

        Returns
        -------
        k : [n_heads, n_valid + n_scratch, d_head]
        v : [n_heads, n_valid + n_scratch, d_head]
        valid_mask : [n_heads, n_valid + n_scratch]  bool
        n_total : int
        """
        n_total = self._n_valid() + self._scratch_count
        return (
            self.k[:, :n_total, :],
            self.v[:, :n_total, :],
            self.valid[:, :n_total],
            n_total,
        )

    def get_positions(self):
        """Return position ids for all valid+scratch slots.

        Returns
        -------
        pos : [n_heads, n_total] int32
        """
        n_total = self._n_valid() + self._scratch_count
        return self.pos[:, :n_total]

    def commit(self, n_keep):
        """Accept the first n_keep scratch tokens, discard the rest.

        Parameters
        ----------
        n_keep : int
            Number of speculative tokens accepted (0 = full rollback,
            scratch_count = accept all).
        """
        if not self._pending_commit:
            raise RuntimeError("commit called without pending begin_block")

        start = self._scratch_start
        n_discard = self._scratch_count - n_keep

        if n_discard > 0:
            # Invalidate and zero out rejected scratch slots
            reject_start = start + n_keep
            reject_end = start + self._scratch_count
            self.valid[:, reject_start:reject_end] = False
            self.score[:, reject_start:reject_end] = 0.0
            self.pos[:, reject_start:reject_end] = -1
            self.k[:, reject_start:reject_end] = 0.0
            self.v[:, reject_start:reject_end] = 0.0

        self.length += n_keep
        self._scratch_count = 0
        self._pending_commit = False
        self._commits_since_evict += 1

    def rollback(self):
        """Reject all scratch tokens (convenience for commit(0))."""
        self.commit(0)

    def update_scores(self, attn_probs, n_keep):
        """Update accumulated attention scores with decay.

        score <- decay * score + new_attn

        For a block of m committed rows, row i is weighted by decay^(m-1-i)
        so that the sequential per-token update is equivalent.

        Parameters
        ----------
        attn_probs : [n_heads, n_keep, n_total_at_forward_time]
            Attention probabilities from the committed tokens.
        n_keep : int
            Number of committed tokens (matches commit's n_keep).
        """
        if n_keep == 0:
            return

        n_total = attn_probs.shape[2]
        # Clamp to our valid range
        n_slots = min(n_total, self._n_valid())

        # For sequential equivalence: row i (0-indexed) gets weight decay^(m-1-i)
        # where m = n_keep.
        if n_keep == 1:
            # Simple case: score <- decay * score + attn[0]
            self.score[:, :n_slots] = (
                self.decay * self.score[:, :n_slots]
                + attn_probs[:, 0, :n_slots].float()
            )
        else:
            # Block update with geometric decay weights
            for i in range(n_keep):
                weight = self.decay ** (n_keep - 1 - i)
                if i == 0:
                    # First row: decay^(m-1) * old_score + attn[0]
                    self.score[:, :n_slots] = (
                        (self.decay ** n_keep) * self.score[:, :n_slots]
                        + weight * attn_probs[:, i, :n_slots].float()
                    )
                else:
                    self.score[:, :n_slots] = (
                        self.score[:, :n_slots]
                        + weight * attn_probs[:, i, :n_slots].float()
                    )

    def evict_to_budget(self):
        """Per-head eviction: drop lowest-score unprotected slots.

        Eviction happens only if:
        1. No pending commit (never evict during speculation)
        2. Number of valid slots > budget
        3. Enough commits since last eviction (evict_every)

        The local window (last n_local positions) is protected.
        The first n_pin positions are protected (attention sinks).
        """
        if self._pending_commit:
            return  # never evict during speculation

        if self._commits_since_evict < self.evict_every:
            return

        n_valid = self._n_valid()
        if n_valid <= self.budget:
            return

        n_to_evict = n_valid - self.budget

        # Per-head eviction
        for h in range(self.n_heads):
            # Get valid slots for this head
            valid_mask_h = self.valid[h, :n_valid].clone()
            positions_h = self.pos[h, :n_valid].clone()
            scores_h = self.score[h, :n_valid].clone()

            # Find the maximum position to identify local window
            max_pos = positions_h[valid_mask_h].max().item() if valid_mask_h.any() else 0
            local_start = max_pos - self.n_local + 1

            # Build protection mask
            protected = torch.zeros(n_valid, device=self.device, dtype=torch.bool)

            # Protect pinned (first n_pin absolute positions)
            for i in range(n_valid):
                if valid_mask_h[i]:
                    p = positions_h[i].item()
                    if p < self.n_pin:
                        protected[i] = True
                    elif p >= local_start:
                        protected[i] = True

            # Find evictable slots
            evictable = valid_mask_h & ~protected
            n_evictable = evictable.sum().item()

            if n_evictable == 0:
                continue

            actual_evict = min(n_to_evict, n_evictable)

            # Get scores of evictable slots
            evictable_indices = torch.where(evictable)[0]
            evictable_scores = scores_h[evictable_indices]

            # Find the lowest-scoring slots
            _, sorted_idx = evictable_scores.sort()
            evict_indices = evictable_indices[sorted_idx[:actual_evict]]

            # Invalidate evicted slots
            self.valid[h, evict_indices] = False
            self.score[h, evict_indices] = 0.0
            self.pos[h, evict_indices] = -1

        # Compact the buffer (move valid slots to front)
        self._compact()
        self._commits_since_evict = 0

    def _compact(self):
        """Compact the buffer per-head so all valid slots are contiguous at 0..n_valid-1.

        After per-head eviction, each head drops its lowest-scoring unprotected tokens,
        creating gaps. We compact each head independently into the front of its buffer.
        """
        for h in range(self.n_heads):
            valid_idx = torch.where(self.valid[h])[0]
            n_v = len(valid_idx)
            if n_v == 0:
                continue
            # If already contiguous at 0..n_v-1, nothing to do for this head
            if valid_idx[-1].item() == n_v - 1:
                continue

            k_compact = self.k[h, valid_idx].clone()
            v_compact = self.v[h, valid_idx].clone()
            pos_compact = self.pos[h, valid_idx].clone()
            score_compact = self.score[h, valid_idx].clone()

            self.k[h, :n_v] = k_compact
            self.v[h, :n_v] = v_compact
            self.pos[h, :n_v] = pos_compact
            self.score[h, :n_v] = score_compact
            self.valid[h, :n_v] = True

            # Clear trailing slots
            self.k[h, n_v:] = 0.0
            self.v[h, n_v:] = 0.0
            self.pos[h, n_v:] = -1
            self.score[h, n_v:] = 0.0
            self.valid[h, n_v:] = False

    # -----------------------------------------------------------------------
    #  Prefill support
    # -----------------------------------------------------------------------

    def prefill_append(self, k_new, v_new, positions):
        """Append KV from prefill (no scratch, no commit needed).

        Parameters
        ----------
        k_new : [n_heads, seq_len, d_head]
        v_new : [n_heads, seq_len, d_head]
        positions : [seq_len]  int tensor
        """
        n_new = k_new.shape[1]
        start = self._n_valid()
        end = start + n_new

        if end > self.capacity:
            raise RuntimeError(
                "H2OCache overflow during prefill: %d + %d > %d" %
                (start, n_new, self.capacity)
            )

        self.k[:, start:end, :] = k_new
        self.v[:, start:end, :] = v_new
        for h in range(self.n_heads):
            self.pos[h, start:end] = positions.to(torch.int32)
        self.valid[:, start:end] = True
        self.length += n_new

    def init_from_prefill(self, k_full, v_full, positions, score_accum):
        """Initialize cache directly from prefill, compressing to budget if needed.

        Parameters
        ----------
        k_full : [n_heads, seq_len, d_head]
        v_full : [n_heads, seq_len, d_head]
        positions : [seq_len] int tensor
        score_accum : [n_heads, seq_len] float tensor
        """
        self.reset()
        seq_len = k_full.shape[1]

        if seq_len <= self.budget:
            self.k[:, :seq_len, :] = k_full
            self.v[:, :seq_len, :] = v_full
            for h in range(self.n_heads):
                self.pos[h, :seq_len] = positions.to(torch.int32)
            self.score[:, :seq_len] = score_accum.float()
            self.valid[:, :seq_len] = True
            self.length = seq_len
            return

        # More tokens than budget -> compress per-head
        max_pos = int(positions.max().item())
        local_start = max_pos - self.n_local + 1

        for h in range(self.n_heads):
            scores_h = score_accum[h].clone()

            # Identify pin and local positions
            pin_mask = positions < self.n_pin
            local_mask = positions >= local_start
            protected_mask = pin_mask | local_mask

            # Candidate heavy hitters
            candidate_mask = ~protected_mask
            candidate_indices = torch.where(candidate_mask)[0]

            n_avail_hh = len(candidate_indices)
            actual_hh = min(self.n_hh, n_avail_hh)

            if actual_hh > 0:
                cand_scores = scores_h[candidate_indices]
                _, topk_idx = torch.topk(cand_scores, actual_hh)
                selected_hh_indices = candidate_indices[topk_idx]
            else:
                selected_hh_indices = torch.tensor([], dtype=torch.long, device=self.device)

            kept_indices = torch.cat([
                torch.where(pin_mask)[0],
                selected_hh_indices,
                torch.where(local_mask)[0],
            ])
            # Sort kept indices by sequence position so positions are monotonic
            kept_indices, _ = kept_indices.sort()

            n_kept = len(kept_indices)
            self.k[h, :n_kept, :] = k_full[h, kept_indices, :]
            self.v[h, :n_kept, :] = v_full[h, kept_indices, :]
            self.pos[h, :n_kept] = positions[kept_indices].to(torch.int32)
            self.score[h, :n_kept] = scores_h[kept_indices]
            self.valid[h, :n_kept] = True

        self.length = seq_len

    def compress_after_prefill(self, attn_probs_sum):
        """Compress the cache to budget after prefill.

        Called after the full-attention prefill of one layer.  Uses the
        accumulated attention probabilities to select heavy hitters.

        Parameters
        ----------
        attn_probs_sum : [n_heads, seq_len]
            Sum of attention probabilities for each position across all
            query positions in the prefill.  This serves as the initial score.
        """
        n_valid = self._n_valid()
        if n_valid <= self.budget:
            return  # already within budget

        # Set initial scores
        self.score[:, :n_valid] = attn_probs_sum[:, :n_valid].float()

        # Now do per-head eviction to budget
        n_to_evict = n_valid - self.budget

        for h in range(self.n_heads):
            positions_h = self.pos[h, :n_valid]
            scores_h = self.score[h, :n_valid].clone()

            max_pos = positions_h.max().item()
            local_start = max_pos - self.n_local + 1

            # Protection mask
            protected = torch.zeros(n_valid, device=self.device, dtype=torch.bool)
            for i in range(n_valid):
                p = positions_h[i].item()
                if p < self.n_pin:
                    protected[i] = True
                elif p >= local_start:
                    protected[i] = True

            evictable = self.valid[h, :n_valid] & ~protected
            n_evictable = evictable.sum().item()
            actual_evict = min(n_to_evict, n_evictable)

            if actual_evict > 0:
                evictable_indices = torch.where(evictable)[0]
                evictable_scores = scores_h[evictable_indices]
                _, sorted_idx = evictable_scores.sort()
                evict_indices = evictable_indices[sorted_idx[:actual_evict]]

                self.valid[h, evict_indices] = False
                self.score[h, evict_indices] = 0.0
                self.pos[h, evict_indices] = -1

        self._compact()

    # -----------------------------------------------------------------------
    #  Queries and stats
    # -----------------------------------------------------------------------

    def stats(self):
        """Return cache statistics for telemetry.

        Returns
        -------
        dict
        """
        n_valid = self._n_valid()
        elem_bytes = 2 if self.dtype == torch.float16 else 4
        kv_bytes = 2 * self.n_heads * n_valid * self.d_head * elem_bytes

        # Score histogram (for UI heatmap)
        scores_valid = []
        for h in range(self.n_heads):
            mask = self.valid[h, :n_valid]
            if mask.any():
                scores_valid.append(self.score[h, :n_valid][mask])

        score_stats = {}
        if scores_valid:
            all_scores = torch.cat(scores_valid)
            score_stats = {
                "min": all_scores.min().item(),
                "max": all_scores.max().item(),
                "mean": all_scores.mean().item(),
                "median": all_scores.median().item(),
            }

        return {
            "n_valid": n_valid,
            "budget": self.budget,
            "capacity": self.capacity,
            "occupancy_pct": 100.0 * n_valid / self.budget if self.budget > 0 else 0,
            "kv_bytes": kv_bytes,
            "kv_mib": kv_bytes / (1024 * 1024),
            "length": self.length,
            "decay": self.decay,
            "n_pin": self.n_pin,
            "n_hh": self.n_hh,
            "n_local": self.n_local,
            "score_stats": score_stats,
        }

    def memory_bytes(self):
        """Return total bytes used by all cache tensors."""
        elem_bytes = 2 if self.dtype == torch.float16 else 4
        kv = 2 * self.n_heads * self.capacity * self.d_head * elem_bytes
        pos_bytes = self.n_heads * self.capacity * 4  # int32
        score_bytes = self.n_heads * self.capacity * 4  # float32
        valid_bytes = self.n_heads * self.capacity  # bool = 1 byte
        return kv + pos_bytes + score_bytes + valid_bytes

    def reset(self):
        """Clear the cache."""
        self.k.zero_()
        self.v.zero_()
        self.pos.fill_(-1)
        self.score.zero_()
        self.valid.zero_()
        self.length = 0
        self._scratch_start = 0
        self._scratch_count = 0
        self._pending_commit = False
        self._commits_since_evict = 0


def create_h2o_caches(n_layers, n_heads, d_head, n_hh=128, n_local=128,
                       n_pin=4, n_scratch=8, decay=0.98, evict_every=1,
                       device='cuda', dtype=torch.float16):
    """Create a list of H2OCache instances, one per layer.

    Parameters
    ----------
    n_layers : int
    n_heads : int
    d_head : int
    n_hh, n_local, n_pin, n_scratch, decay, evict_every : see H2OCache
    device : str
    dtype : torch.dtype

    Returns
    -------
    list of H2OCache
    """
    return [
        H2OCache(
            n_heads=n_heads, d_head=d_head,
            n_hh=n_hh, n_local=n_local,
            n_pin=n_pin, n_scratch=n_scratch,
            decay=decay, evict_every=evict_every,
            device=device, dtype=dtype,
        )
        for _ in range(n_layers)
    ]
