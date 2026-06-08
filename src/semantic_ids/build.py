"""Turn raw per-item codes into the token space used by the seq2seq model.

Responsibilities
----------------
1. **Collision resolution.** Items that receive identical content codes get an
   extra disambiguation token so every item has a unique Semantic ID. We record
   how often this is needed (collision rate) -- a key diagnostic for RQ2,
   because the extra token is *not* semantic. For temporal experiments, callers
   can give train-window items priority so future/cold catalogue items cannot
   change the disambiguation token assigned to items seen during training.
2. **Per-position offsets.** Code value v at position p becomes a distinct token
   id (``offset + p * codebook_size + v``). Without this, "3" at level 1 and
   "3" at level 2 would collide in the embedding table (TIGER uses per-level
   tokens for the same reason).
3. **Token-order variants** (RQ2): the first L content codes can be kept in the
   original order, reversed, or randomly permuted before tokenisation. The
   disambiguation token is last by default, or first for the ``collision_first``
   ablation.
4. **Decoding trie**: enables prefix-constrained beam search (only valid
   Semantic IDs can be generated) and exact token->item decoding.

Special token ids: PAD=0, EOS=1, BOS=2, then optional user-hash tokens, then
the semantic tokens. ``num_user_tokens=0`` disables the user token.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

PAD, EOS, BOS = 0, 1, 2


class SemanticIDSpace:
    def __init__(
        self,
        raw_codes: np.ndarray,         # (N, L) content codes
        codebook_size: int,
        token_order: str = "original",  # original | reversed | permuted
        num_user_tokens: int = 2000,
        num_gap_tokens: int = 0,        # >0 enables time-gap tokens (RQ3)
        collision_first: bool = False,  # place disambiguation token first (RQ2)
        disambiguation_priority=None,   # optional 1-based item ids assigned first
        seed: int = 42,
    ):
        self.codebook_size = codebook_size
        self.num_user_tokens = num_user_tokens
        self.num_gap_tokens = num_gap_tokens
        self.collision_first = collision_first
        raw_codes = np.asarray(raw_codes)
        if raw_codes.ndim != 2 or raw_codes.shape[0] == 0 or raw_codes.shape[1] == 0:
            raise ValueError("raw_codes must be a non-empty 2D array of shape (num_items, num_levels)")
        if not np.issubdtype(raw_codes.dtype, np.integer):
            raise ValueError("raw_codes must contain integer codeword ids")
        if np.any(raw_codes < 0) or np.any(raw_codes >= codebook_size):
            lo, hi = int(raw_codes.min()), int(raw_codes.max())
            raise ValueError(
                f"raw_codes values must be in [0, {codebook_size}); got min={lo}, max={hi}"
            )

        self.rng = np.random.default_rng(seed)
        L = raw_codes.shape[1]

        # ---- token-order transform on the content codes ----
        perm = list(range(L))
        if token_order == "reversed":
            perm = perm[::-1]
        elif token_order == "permuted":
            perm = list(self.rng.permutation(L))
        self.perm = perm
        codes = raw_codes[:, perm]
        self.content_codes = codes        # (N, L) content codes BEFORE disambiguation
                                          # token; used for the pre-disambiguation
                                          # collision metric regardless of layout.

        # ---- collision-resolving token ----
        seen = defaultdict(int)
        dis = np.zeros(len(codes), dtype=np.int64)
        order = self._disambiguation_order(len(codes), disambiguation_priority)
        for i in order:
            c = tuple(codes[i])
            dis[i] = seen[c]
            seen[c] += 1
        if disambiguation_priority is None:
            self.disambiguation_priority_size = 0
        else:
            self.disambiguation_priority_size = len({
                int(i) for i in disambiguation_priority
                if 1 <= int(i) <= len(codes)
            })
        self.max_collisions = int(dis.max()) + 1
        # The paper uses one full codebook-sized vocabulary slice for each
        # Semantic-ID position, including the appended collision/disambiguation
        # token. Keeping only ``max_collisions`` tokens makes the reported
        # generation task easier and changes invalid-code diagnostics. Use at
        # least ``codebook_size`` slots, while still supporting pathological
        # datasets where one content code collides more than K times.
        self.disambiguation_size = max(codebook_size, self.max_collisions)
        prefix_counts = Counter(map(tuple, codes))
        self.collision_rate = sum(v for v in prefix_counts.values() if v > 1) / len(codes)

        # full code per item = content codes + disambiguation index
        if collision_first:
            self.full_codes = np.concatenate([dis[:, None], codes], axis=1)
            self.pos_sizes = [self.disambiguation_size] + [codebook_size] * L
        else:
            self.full_codes = np.concatenate([codes, dis[:, None]], axis=1)  # (N, L+1)
            self.pos_sizes = [codebook_size] * L + [self.disambiguation_size]
        self.code_len = L + 1

        # ---- token-id layout ----
        self.sem_offset = 3 + num_user_tokens
        self.pos_offset = [self.sem_offset]
        for s in self.pos_sizes[:-1]:
            self.pos_offset.append(self.pos_offset[-1] + s)
        self.gap_offset = self.pos_offset[-1] + self.pos_sizes[-1]
        self.vocab_size = self.gap_offset + num_gap_tokens

        # ---- mappings ----
        self.item_tokens = np.stack(
            [self._code_to_tokens(c) for c in self.full_codes]
        )  # (N, L+1) token ids
        # NOTE: item ids are 1-based everywhere (id 0 is reserved for padding in
        # dataset.py / SASRec). full_codes row i therefore corresponds to item
        # id i+1, so the code->item map MUST be 1-based or generative evaluation
        # silently compares the wrong ids.
        self.code_to_item = {tuple(c): i + 1 for i, c in enumerate(self.full_codes)}
        self.valid_codes = set(self.code_to_item.keys())
        self._build_trie()

    # -- token helpers -------------------------------------------------------
    @staticmethod
    def _disambiguation_order(num_items: int, priority_item_ids):
        """0-based item row order used when assigning collision tokens.

        Default TIGER assigns disambiguation indices in catalogue order. In a
        temporal split, however, assigning across the whole catalogue in raw item
        order lets future/cold items with smaller ids shift the target code of
        train-window items. Passing the train-window item ids as priority keeps
        train-item codes identical to the code assignment they would receive if
        future items were absent; remaining catalogue items are then assigned
        deterministically after them.
        """
        if priority_item_ids is None:
            return list(range(num_items))
        priority_rows, seen = [], set()
        for item_id in priority_item_ids:
            item_id = int(item_id)
            if item_id < 1 or item_id > num_items:
                continue
            row = item_id - 1
            if row not in seen:
                priority_rows.append(row)
                seen.add(row)
        tail = [i for i in range(num_items) if i not in seen]
        return priority_rows + tail

    def _code_to_tokens(self, code) -> np.ndarray:
        return np.array([self.pos_offset[p] + int(v) for p, v in enumerate(code)], dtype=np.int64)

    def _tokens_to_code(self, tokens):
        return tuple(int(t) - self.pos_offset[p] for p, t in enumerate(tokens))

    def user_token(self, user_id: int) -> int:
        if self.num_user_tokens == 0:
            return None
        return 3 + (user_id % self.num_user_tokens)

    def gap_token(self, bucket: int) -> int:
        return self.gap_offset + min(bucket, self.num_gap_tokens - 1)

    def tokens_to_item(self, tokens):
        """Map a generated token sequence to an item id, or None if invalid."""
        if len(tokens) != self.code_len:
            return None
        try:
            code = self._tokens_to_code(tokens)
        except Exception:
            return None
        return self.code_to_item.get(code)

    # -- trie for constrained decoding --------------------------------------
    def _build_trie(self):
        self.trie = {}
        for toks in self.item_tokens:
            node = self.trie
            for t in toks:
                node = node.setdefault(int(t), {})
            node[EOS] = {}

    def allowed_tokens(self, prefix):
        """Return the set of token ids allowed after ``prefix`` (for HF
        ``prefix_allowed_tokens_fn``). ``prefix`` is the decoded sequence so far
        *excluding* the decoder-start token."""
        node = self.trie
        for t in prefix:
            if int(t) not in node:
                return [EOS]
            node = node[int(t)]
        keys = list(node.keys())
        return keys if keys else [EOS]
