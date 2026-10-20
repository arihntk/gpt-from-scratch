import argparse
import json
import math
import time
from pathlib import Path

import torch

from config import GptConfig, TrainConfig
from dataset import Loader
from model import Gpt
from tokenizer import Bpe


class Log:
    def __init__(self, out_dir):
        self.file = open(Path(out_dir) / "log.txt", "a")
        self.t0 = time.time()

    def __call__(self, msg):
        stamp = f"[{time.time() - self.t0:8.1f}s]"
        print(stamp + " " + msg, flush=True)
        self.file.write(stamp + " " + msg + "\n")
        self.file.flush()


def lr_at(step, base, warmup, steps, min_ratio):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def param_groups(model, weight_decay):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, loader, iters, batch, block, device):
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = loader.sample(batch, block, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / iters


def main():
    ap = argparse.ArgumentParser(description="train a small gpt on mps")
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
    args = ap.parse_args()

    mc, tc = GptConfig(), TrainConfig()
    for k in vars(args):
        v = vars(args)[k]
        if k in mc.__dataclass_fields__ and v is not None:
            setattr(mc, k, v)
        if k in tc.__dataclass_fields__ and v is not None:
            setattr(tc, k, v)

    device = pick_device()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    log = Log(args.out_dir)
    torch.manual_seed(tc.seed)

    tok = Bpe.load(Path(args.data_dir) / "tok.json")
    mc.vocab_size = tok.vocab_size
    train_loader = Loader(args.data_dir, "train")
    val_loader = Loader(args.data_dir, "val")
    log(f"data {args.data_dir}: {train_loader.n:,} train tokens, {val_loader.n:,} val tokens")

    model = Gpt(mc).to(device)
    log(f"device {device} | model {model.n_params:,} params | cfg {mc.dict()}")
    log(f"sanity: untrained loss should be about ln(vocab) = {math.log(mc.vocab_size):.2f}")

    opt = torch.optim.AdamW(
        param_groups(model, tc.weight_decay), lr=tc.lr, betas=(0.9, 0.95), eps=1e-8
    )

    start_step = 0
    best_val = float("inf")
    last_ckpt = Path(args.out_dir) / "last.pt"
    if args.resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"] + 1
        best_val = state["best_val"]
        log(f"resumed from step {state['step']}, best_val {best_val:.4f}")

    tokens_per_step = tc.batch_size * mc.block_size
    log(f"tokens per step: {tokens_per_step:,} | steps: {tc.steps}")

    history = []
    t_train = time.time()
    for step in range(start_step, tc.steps):
        lr = lr_at(step, tc.lr, tc.warmup, tc.steps, tc.min_lr_ratio)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = train_loader.sample(tc.batch_size, mc.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()

        if step % 10 == 0 or step == tc.steps - 1:
            dt = time.time() - t_train
            t_train = time.time()
            spill = 10 if step else 1
            grad_norm = norm.item() if torch.is_tensor(norm) else float(norm)
            eta = (tc.steps - step - 1) * dt / spill
            log(
                f"step {step:5d}/{tc.steps} | loss {loss.item():.4f} | lr {lr:.2e} "
                f"| grad {grad_norm:.2f} | {spill * tokens_per_step / dt / 1e3:.1f}k tok/s "
                f"| {dt * 1000 / spill:.0f} ms/step | eta {eta / 60:.1f}m"
            )

        is_last = step == tc.steps - 1
        if (step % tc.eval_interval == 0 and step % 10 == 0) or is_last:
            tr = estimate_loss(model, train_loader, tc.eval_iters, tc.batch_size, mc.block_size, device)
            va = estimate_loss(model, val_loader, tc.eval_iters, tc.batch_size, mc.block_size, device)
            log(
                f"eval @ {step:5d} | train {tr:.4f} (ppl {math.exp(tr):.2f}) "
                f"| val {va:.4f} (ppl {math.exp(va):.2f})"
            )
            history.append({"step": step, "train": tr, "val": va})
            if va < best_val:
                best_val = va
                src = model.state_dict()
                clean = {k: v.cpu() for k, v in src.items()}
                torch.save(
                    {
                        "model": clean,
                        "config": mc.dict(),
                        "step": step,
                        "val_loss": va,
                        "data_dir": args.data_dir,
                    },
                    Path(args.out_dir) / "best.pt",
                )
                log(f"saved best.pt at step {step} with val {va:.4f}")

            if step == 0 or is_last:
                model.eval()
                prompt = torch.tensor(
                    [tok.encode("\n")], dtype=torch.long, device=device
                )
                out = model.generate(prompt, args.sample_tokens, temperature=0.8, top_k=64)
                model.train()
                log("sample:\n" + tok.decode(out[0].tolist()))

    torch.save(
        {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "opt": opt.state_dict(),
            "config": mc.dict(),
            "train": {k: getattr(tc, k) for k in tc.__dataclass_fields__},
            "step": tc.steps - 1,
            "best_val": best_val,
            "data_dir": args.data_dir,
        },
        last_ckpt,
    )
    (Path(args.out_dir) / "history.json").write_text(json.dumps(history, indent=2))
    log(f"done. best val {best_val:.4f} | last.pt, best.pt, history.json in {args.out_dir}/")


if __name__ == "__main__":
    main()
