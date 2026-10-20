import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np

from tokenizer import Bpe

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    ap = argparse.ArgumentParser(description="prepare a text corpus for training")
    ap.add_argument("--dir", default="data/shakespeare")
    ap.add_argument("--input", default=None, help="local utf-8 text file to use instead of download")
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    assert args.vocab_size < 65536, "uint16 export requires vocab below 65536"

    raw = Path(args.input) if args.input else out / "input.txt"
    if not raw.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, raw)
    text = raw.read_text(encoding="utf-8")
    n_bytes = len(text.encode("utf-8"))
    print(f"corpus: {raw} | {len(text):,} chars | {n_bytes:,} bytes")

    t0 = time.time()
    tok = Bpe()
    ids = tok.fit(text, args.vocab_size)
    print(f"tokenizer: {tok.vocab_size} vocab entries | {len(ids):,} tokens | {time.time() - t0:.1f}s")
    print(f"compression: {n_bytes / len(ids):.2f} bytes per token")
    tok.save(out / "tok.json")

    all_ids = np.array(ids, dtype=np.uint16)
    n = int((1 - args.val_frac) * len(all_ids))
    all_ids[:n].tofile(out / "train.bin")
    all_ids[n:].tofile(out / "val.bin")
    print(f"split: {n:,} train | {len(all_ids) - n:,} val tokens")
    print(f"wrote {out}/train.bin, {out}/val.bin, {out}/tok.json")


if __name__ == "__main__":
    main()
