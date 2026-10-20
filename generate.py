import argparse
from pathlib import Path

import torch

from config import GptConfig
from model import Gpt
from tokenizer import Bpe


def main():
    ap = argparse.ArgumentParser(description="sample text from a trained checkpoint")
    ap.add_argument("--ckpt", default="run/best.pt")
    ap.add_argument("--data-dir", default=None, help="overrides the dir recorded in the checkpoint")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    mc = GptConfig(**state["config"])
    data_dir = args.data_dir or state["data_dir"]
    tok = Bpe.load(Path(data_dir) / "tok.json")

    model = Gpt(mc).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(
        f"loaded {args.ckpt} | {model.n_params:,} params | step {state['step']} "
        f"| val loss {state['val_loss']:.4f}"
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    for i in range(args.samples):
        out = model.generate(ids, args.max_new, temperature=args.temperature, top_k=args.top_k)
        text = tok.decode(out[0].tolist())
        print(f"\n---------- sample {i + 1} ----------")
        print(text)


if __name__ == "__main__":
    main()
