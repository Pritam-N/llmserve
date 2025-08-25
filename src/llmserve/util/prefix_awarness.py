# SPDX-License-Identifier: Apache-2.0
# TODO: Later we'll replace this with real APC stats from vLLM.
from __future__ import annotations
import hashlib
from collections import OrderedDict

class PrefixHeuristic:
    """
    Lightweight LRU frequency estimator for prefix reuse. Until we tap vLLM's
    Automatic Prefix Caching stats directly, we approximate hit probability by
    seeing how often a prefix hash reappears.
    """
    def __init__(self, max_entries: int = 4096):
        self.max = max_entries
        self.lru: OrderedDict[str, int] = OrderedDict()

    @staticmethod
    def _prefix_key(prompt: str, first_n_chars: int = 512) -> str:
        # Hash first N chars (system + first user turn) as a prefix signature
        p = prompt[:first_n_chars].encode("utf-8", errors="ignore")
        return hashlib.blake2b(p, digest_size=16).hexdigest()

    def observe(self, prompt: str) -> float:
        """
        Record the prefix and return an estimated hit probability in [0,1].
        """
        key = self._prefix_key(prompt)
        cnt = self.lru.get(key, 0) + 1
        if key in self.lru:
            self.lru.move_to_end(key)
        self.lru[key] = cnt
        # evict if needed
        if len(self.lru) > self.max:
            self.lru.popitem(last=False)
        # simple mapping: >=3 sightings -> high probability
        if cnt >= 5: return 0.9
        if cnt == 4: return 0.7
        if cnt == 3: return 0.5
        if cnt == 2: return 0.3
        return 0.1