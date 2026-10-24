# -*- coding: utf-8 -*-
"""H2O-weighted suffix-match drafter (P3-1).

Zero-RAM drafter: keeps only committed token IDs (ints) and an incremental
n-gram index.  No KV cache, no model weights — the ~MB cost is the "c ~= 0"
claim from the spec-decode paper.

Candidate selection:
1. **Importance-weighted** (default): weight each candidate continuation by
   its H2O importance score.
2. **Most-recent-match** baseline: always pick the most recent match.
3. **Longest-match** baseline: prefer the longest matching suffix.

These two baselines approximate prompt-lookup approaches and serve as the
honest comparison for the novelty claim.

Python-3.6 safe.
"""

import collections


class NgramIndex(object):
    """Incremental n-gram index over a token-ID sequence.

    Maintains inverted lists for n-grams of length 1..max_n.  When a new
    token is committed, all n-grams ending at that position are registered.

    Parameters
    ----------
    max_n : int
        Maximum n-gram length to index (default 4).
    """

    def __init__(self, max_n=4):
        self.max_n = max_n
        # ngram_to_positions[n][tuple_of_ids] -> list of ending positions
        self.ngram_to_positions = {}
        for n in range(1, max_n + 1):
            self.ngram_to_positions[n] = collections.defaultdict(list)

    def add_token(self, history, pos):
        """Register n-grams ending at *pos* in *history*.

        Parameters
        ----------
        history : list of int
            Full token-ID history (len >= pos+1).
        pos : int
            Index of the newly committed token in *history*.
        """
        for n in range(1, self.max_n + 1):
            start = pos - n + 1
            if start < 0:
                continue
            key = tuple(history[start:pos + 1])
            self.ngram_to_positions[n][key].append(pos)

    def lookup(self, suffix, n):
        """Find all positions whose preceding n tokens match *suffix*.

        Parameters
        ----------
        suffix : tuple of int, length n
        n : int

        Returns
        -------
        list of int
            Positions where the n-gram ends.
        """
        return self.ngram_to_positions.get(n, {}).get(suffix, [])

    def reset(self):
        """Clear all indexed n-grams."""
        for n in range(1, self.max_n + 1):
            self.ngram_to_positions[n] = collections.defaultdict(list)


class SuffixDrafter(object):
    """H2O-importance-weighted suffix-match drafter.

    Parameters
    ----------
    max_n : int
        Maximum n-gram length for suffix matching (1..4).
    max_history : int
        Maximum token-ID history length (default 2048).
    selector : str
        One of 'importance' (default), 'most_recent', 'longest'.
    """

    def __init__(self, max_n=4, max_history=2048, selector='importance'):
        self.max_n = max_n
        self.max_history = max_history
        self.selector = selector

        self.history = []          # all committed token IDs
        self.importance = {}       # position -> importance score (from H2O)
        self.index = NgramIndex(max_n=max_n)

    def reset(self):
        """Clear drafter state."""
        self.history = []
        self.importance = {}
        self.index.reset()

    def commit_tokens(self, token_ids, importance_scores=None):
        """Register newly committed tokens.

        Parameters
        ----------
        token_ids : list of int
            Token IDs that were just committed.
        importance_scores : list of float or None
            H2O importance scores for the committed positions.
            If None, defaults to 1.0 for each.
        """
        if importance_scores is None:
            importance_scores = [1.0] * len(token_ids)

        for tid, imp in zip(token_ids, importance_scores):
            pos = len(self.history)
            self.history.append(tid)
            self.importance[pos] = imp
            self.index.add_token(self.history, pos)

    def update_importance(self, position, score):
        """Update the importance score for a given position.

        Parameters
        ----------
        position : int
        score : float
        """
        self.importance[position] = score

    def draft(self, gamma):
        """Produce up to *gamma* draft token IDs.

        The drafter looks for the longest suffix match (back off from
        max_n to 1), then extends the match to produce up to gamma
        continuation tokens.

        Parameters
        ----------
        gamma : int
            Maximum number of draft tokens to produce.

        Returns
        -------
        draft_ids : list of int
            Draft token IDs (0 to gamma tokens).
        draft_probs : list of dict
            For each draft position, a dict {token_id: probability} representing
            the draft distribution q.  For greedy mode, this is a one-hot.
        match_length : int
            Length of the suffix that matched (0 if no match).
        """
        if len(self.history) < 1 or gamma == 0:
            return [], [], 0

        draft_ids = []
        draft_probs = []
        match_length = 0

        # Try suffix lengths max_n down to 1
        best_matches = None
        for n in range(min(self.max_n, len(self.history)), 0, -1):
            suffix = tuple(self.history[-n:])
            positions = self.index.lookup(suffix, n)

            # Filter: only positions that have a continuation AND are not
            # the current tail (to avoid self-match)
            candidates = []
            for p in positions:
                # p is the position where the n-gram ends
                # Continuation starts at p+1
                if p + 1 < len(self.history) and p != len(self.history) - 1:
                    candidates.append(p)

            if candidates:
                best_matches = candidates
                match_length = n
                break

        if best_matches is None:
            return [], [], 0

        # Select candidate and extend
        if self.selector == 'most_recent':
            draft_ids, draft_probs = self._select_most_recent(
                best_matches, gamma
            )
        elif self.selector == 'longest':
            draft_ids, draft_probs = self._select_longest(
                best_matches, gamma
            )
        else:
            # Default: importance-weighted
            draft_ids, draft_probs = self._select_importance_weighted(
                best_matches, gamma
            )

        return draft_ids, draft_probs, match_length

    def _select_importance_weighted(self, candidates, gamma):
        """Importance-weighted candidate selection.

        Each candidate continuation is weighted by the H2O importance score
        of the continuation token's position.  For greedy mode, pick the
        highest-weight continuation.  For sampling, build a normalised
        distribution q(x) over next-token IDs from the weight mass.

        Returns
        -------
        draft_ids : list of int
        draft_probs : list of dict
        """
        draft_ids = []
        draft_probs = []

        # For each draft position, collect candidate tokens with weights
        current_candidates = list(candidates)

        for step in range(gamma):
            if not current_candidates:
                break

            # Collect continuations: token_id -> total importance weight
            token_weights = collections.defaultdict(float)
            token_sources = collections.defaultdict(list)  # token_id -> list of (pos,)

            for pos in current_candidates:
                cont_pos = pos + 1 + step
                if cont_pos >= len(self.history):
                    continue
                cont_token = self.history[cont_pos]
                weight = self.importance.get(cont_pos, 1.0)
                token_weights[cont_token] += weight
                token_sources[cont_token].append(pos)

            if not token_weights:
                break

            # Normalise to get q(x)
            total = sum(token_weights.values())
            if total <= 0:
                total = 1.0
            q = {}
            for tid, w in token_weights.items():
                q[tid] = w / total

            # Greedy: pick highest weight
            best_token = max(token_weights, key=token_weights.get)
            draft_ids.append(best_token)
            draft_probs.append(q)

            # Filter candidates to those whose continuation at this step
            # matches the chosen token
            next_candidates = []
            for pos in current_candidates:
                cont_pos = pos + 1 + step
                if cont_pos < len(self.history) and self.history[cont_pos] == best_token:
                    next_candidates.append(pos)
            current_candidates = next_candidates

        return draft_ids, draft_probs

    def _select_most_recent(self, candidates, gamma):
        """Most-recent-match baseline: always choose the last occurrence."""
        # Pick the candidate with the highest position (most recent)
        best_pos = max(candidates)
        draft_ids = []
        draft_probs = []

        for step in range(gamma):
            cont_pos = best_pos + 1 + step
            if cont_pos >= len(self.history):
                break
            tid = self.history[cont_pos]
            draft_ids.append(tid)
            draft_probs.append({tid: 1.0})

        return draft_ids, draft_probs

    def _select_longest(self, candidates, gamma):
        """Longest-match baseline: prefer the candidate with longest
        continuation matching the history suffix."""
        # For each candidate, see how far the continuation extends
        # as a match from the current position
        best_pos = candidates[0]
        best_run = 0

        for pos in candidates:
            run = 0
            for step in range(gamma):
                cont_pos = pos + 1 + step
                if cont_pos >= len(self.history):
                    break
                run += 1
            if run > best_run:
                best_run = run
                best_pos = pos

        draft_ids = []
        draft_probs = []
        for step in range(gamma):
            cont_pos = best_pos + 1 + step
            if cont_pos >= len(self.history):
                break
            tid = self.history[cont_pos]
            draft_ids.append(tid)
            draft_probs.append({tid: 1.0})

        return draft_ids, draft_probs

    def get_draft_distribution(self, candidates, step):
        """Build the normalised draft distribution q(x) for a given step.

        This is used by the sampling-mode speculative sampler so it has
        the exact q used in the accept/reject rule.

        Parameters
        ----------
        candidates : list of int
            Candidate positions from suffix matching.
        step : int
            Draft step index (0-based).

        Returns
        -------
        dict : {token_id: float}
        """
        token_weights = collections.defaultdict(float)
        for pos in candidates:
            cont_pos = pos + 1 + step
            if cont_pos >= len(self.history):
                continue
            cont_token = self.history[cont_pos]
            weight = self.importance.get(cont_pos, 1.0)
            token_weights[cont_token] += weight

        total = sum(token_weights.values())
        if total <= 0:
            return {}
        return {tid: w / total for tid, w in token_weights.items()}
