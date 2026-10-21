import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_model

from config import GptConfig, pick_device
from model import Gpt
from tokenizer import Bpe


def load(path, device):
    path = Path(path)
    if path.suffix == ".safetensors":
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {"config": json.loads((path.parent / "config.json").read_text())}
            extra_path = path.parent / "metadata.json"
            if extra_path.exists():
                extra = json.loads(extra_path.read_text())
                meta.update({k: v for k, v in extra.items() if k != "config"})
        model = Gpt(GptConfig(**meta["config"])).to(device)
        load_model(model, path)
        return model, meta
    state = torch.load(path, map_location="cpu", weights_only=True)
    meta = {k: v for k, v in state.items() if k != "model"}
    model = Gpt(GptConfig(**meta["config"])).to(device)
    model.load_state_dict(state["model"])
    return model, meta


def main():
    ap = argparse.ArgumentParser(description="sample from a trained gpt")
    ap.add_argument("--ckpt", default=None, help="default run/best.safetensors, then run/best.pt")
    ap.add_argument("--data-dir", default=None, help="override the dir recorded in the checkpoint")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else Path("run/best.safetensors")
    if args.ckpt is None and not ckpt.exists():
        ckpt = Path("run/best.pt")
    dev = pick_device()
    model, meta = load(ckpt, dev)
    data_dir = Path(args.data_dir or meta.get("data_dir") or ".")
    tok = Bpe.load(data_dir / "tok.json")
    model.eval()
    line = f"loaded {ckpt} | {model.n_params:,} params"
    if meta.get("step") is not None:
        line += f" | step {meta['step']}"
    if meta.get("val_loss") is not None:
        line += f" | val loss {meta['val_loss']:.4f}"
    print(line)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=dev)
    for i in range(args.samples):
        out = model.generate(ids, args.max_new, temperature=args.temperature, top_k=args.top_k)
        print(f"\n---------- sample {i + 1} ----------\n" + tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
