import argparse
import json
import math
import time
from pathlib import Path

import torch
from safetensors.torch import save_model

from config import GptConfig, TrainConfig, pick_device
from dataset import Loader
from model import Gpt
from tokenizer import Bpe


def make_log(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file = open(out_dir / "log.txt", "a")
    opened = time.time()

    def log(msg):
        line = f"[{time.time() - opened:8.1f}s] {msg}"
        print(line, flush=True)
        file.write(line + "\n")
        file.flush()

    return log


def lr_at(step, tc):
    if step < tc.warmup:
        return tc.lr * (step + 1) / tc.warmup
    p = (step - tc.warmup) / max(1, tc.steps - tc.warmup)
    return tc.lr * (tc.min_lr_ratio + (1 - tc.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def save(model, path, step, data_dir, opt=None, **extra):
    path = Path(path)
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    meta = {
        "config": model.cfg.dict(),
        "step": step,
        "data_dir": data_dir,
        "params": model.n_params,
        "torch": str(torch.__version__),
        **extra,
    }
    body = {"model": weights, **meta}
    if opt is not None:
        body["opt"] = opt.state_dict()
    torch.save(body, path)
    cpu = Gpt(model.cfg)
    cpu.load_state_dict(weights)
    save_model(cpu, path.with_suffix(".safetensors"))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def sample_text(model, tok, n):
    dev = next(model.parameters()).device
    ids = torch.tensor([tok.encode("\n")], dtype=torch.long, device=dev)
    out = model.generate(ids, n, temperature=0.8, top_k=64)
    model.train()
    return tok.decode(out[0].tolist())


def flags():
    ap = argparse.ArgumentParser(description="train a small gpt")
    ap.add_argument("--data-dir", default="data/shakespeare")
    ap.add_argument("--out-dir", default="run")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--block-size", type=int)
    ap.add_argument("--n-layer", type=int)
    ap.add_argument("--n-embd", type=int)
    ap.add_argument("--n-head", type=int)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--grad-clip", type=float)
    ap.add_argument("--eval-interval", type=int)
    ap.add_argument("--eval-iters", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sample-tokens", type=int, default=200)
    return ap.parse_args()


def main():
    args = flags()
    mc, tc = GptConfig(), TrainConfig()
    for key, val in vars(args).items():
        if val is None:
            continue
        if key in mc.__dataclass_fields__:
            setattr(mc, key, val)
        elif key in tc.__dataclass_fields__:
            setattr(tc, key, val)

    log = make_log(args.out_dir)
    torch.manual_seed(tc.seed)

    tok = Bpe.load(Path(args.data_dir) / "tok.json")
    mc.vocab_size = tok.vocab_size
    loaders = {"train": Loader(args.data_dir, "train"), "val": Loader(args.data_dir, "val")}
    log(f"data {args.data_dir}: {loaders['train'].n:,} train, {loaders['val'].n:,} val tokens")

    dev = pick_device()
    model = Gpt(mc).to(dev)
    log(f"device {dev} | {model.n_params:,} params | {mc.dict()}")
    log(f"sanity: untrained loss should be near ln(vocab) = {math.log(mc.vocab_size):.2f}")

    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": tc.weight_decay},
            {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        lr=tc.lr,
    )

    start = 0
    best = float("inf")
    history = []
    resumed = False
    if args.resume and (Path(args.out_dir) / "last.pt").exists():
        st = torch.load(Path(args.out_dir) / "last.pt", map_location=dev, weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["step"] + 1
        best = st["best_val"]
        resumed = True
        log(f"resumed at step {start}, best val {best:.4f}")
    if start >= tc.steps:
        log(f"nothing to train: start {start} >= steps {tc.steps}")
        return
    if resumed and (Path(args.out_dir) / "history.json").exists():
        history = json.loads((Path(args.out_dir) / "history.json").read_text())

    log(f"tokens per step: {tc.batch_size * mc.block_size:,} | steps: {tc.steps}")

    def score(loader, iters):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for _ in range(iters):
                x, y = loader.sample(tc.batch_size, mc.block_size, dev)
                _, loss = model(x, y)
                total += loss.item()
        model.train()
        return total / iters

    timer = time.time()
    recent = 0
    for step in range(start, tc.steps):
        lr = lr_at(step, tc)
        for group in opt.param_groups:
            group["lr"] = lr

        x, y = loaders["train"].sample(tc.batch_size, mc.block_size, dev)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        recent += 1

        if step % 10 == 0 or step == tc.steps - 1:
            per = (time.time() - timer) / recent
            timer = time.time()
            recent = 0
            log(
                f"step {step:5d}/{tc.steps} | loss {loss.item():.4f} | lr {lr:.2e} "
                f"| grad {norm:.2f} | {per:.2f} s/step"
                f" | {tc.batch_size * mc.block_size / per / 1e3:.1f}k tok/s"
                f" | eta {(tc.steps - step - 1) * per / 60:.1f}m"
            )

        if step % tc.eval_interval == 0 or step == tc.steps - 1:
            tr = score(loaders["train"], tc.eval_iters)
            va = score(loaders["val"], tc.eval_iters)
            log(
                f"eval @ {step:5d} | train {tr:.4f} (ppl {math.exp(tr):.2f})"
                f" | val {va:.4f} (ppl {math.exp(va):.2f})"
            )
            history.append({"step": step, "train": tr, "val": va})
            improved = va < best
            if improved:
                best = va
                save(model, Path(args.out_dir) / "best.pt", step, args.data_dir, val_loss=va)
                text = sample_text(model, tok, args.sample_tokens)
                (Path(args.out_dir) / "best_sample.txt").write_text(
                    f"step {step} | val {va:.4f}\n\n{text}\n"
                )
                log(f"saved best.pt at step {step} with val {va:.4f}")
                log("sample @ best:\n" + text)
            if step == tc.steps - 1 and not improved:
                log("sample:\n" + sample_text(model, tok, args.sample_tokens))

    save(
        model,
        Path(args.out_dir) / "last.pt",
        tc.steps - 1,
        args.data_dir,
        opt=opt,
        best_val=best,
        train={k: getattr(tc, k) for k in tc.__dataclass_fields__},
    )
    (Path(args.out_dir) / "history.json").write_text(json.dumps(history, indent=2))
    log(f"done | best val {best:.4f} | best.pt, last.pt, history.json in {args.out_dir}/")


if __name__ == "__main__":
    main()
