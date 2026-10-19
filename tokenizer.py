import json
from collections import Counter
from pathlib import Path


class Bpe:
    def __init__(self):
        self.merges = {}
        self.vocab = [bytes([i]) for i in range(256)]

    def fit(self, text, vocab_size):
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            stats = Counter(zip(ids, ids[1:]))
            if not stats:
                print(f"ran out of pairs after {i} merges")
                break
            pair = stats.most_common(1)[0][0]
            nxt = 256 + i
            ids = self._apply(ids, pair, nxt)
            self.merges[pair] = nxt
            if (i + 1) % 128 == 0:
                print(f"merge {i + 1:4d}: {pair} -> {nxt} | tokens {len(ids):,}")
        self._build_vocab()
        return ids

    @staticmethod
    def _apply(ids, pair, idx):
        out = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(idx)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _build_vocab(self):
        vocab = [bytes([i]) for i in range(256)]
        for pair, idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab.append(vocab[pair[0]] + vocab[pair[1]])
        self.vocab = vocab

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = Counter(zip(ids, ids[1:]))
            pair = min(stats, key=lambda p: self.merges.get(p, 1 << 30))
            if pair not in self.merges:
                break
            ids = self._apply(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        body = {
            "vocab_size": self.vocab_size,
            "merges": [
                [a, b, idx]
                for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1])
            ],
        }
        Path(path).write_text(json.dumps(body))

    @classmethod
    def load(cls, path):
        tok = cls()
        for a, b, idx in json.loads(Path(path).read_text())["merges"]:
            tok.merges[(a, b)] = idx
        tok._build_vocab()
        return tok
