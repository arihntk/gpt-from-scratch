import argparse
import math
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi

from config import GptConfig
from model import Gpt

INFER_FILES = ["config.py", "model.py", "tokenizer.py", "generate.py"]

CARD = """---
tags:
  - gpt2
  - text-generation
  - shakespeare
  - pytorch
  - tiny
---

# gpt2 shakespeare

## Summary

A small GPT2 style language model with {params:,} parameters, implemented and trained from scratch in PyTorch. The tokenizer is a byte level BPE model with a {vocab} token vocabulary trained on the same corpus.

## Status

This checkpoint was saved at step {step} of a {planned} step training plan. Validation loss at this point is {loss:.4f} and validation perplexity is {ppl:.2f}.

## Files

- best.pt: model weights and training metadata
- tok.json: BPE merge table
- config.py, model.py, tokenizer.py, generate.py: the code needed to run the model
- README.md: this model card

## Usage

```
pip install torch numpy
python generate.py --ckpt best.pt --data-dir . --prompt "ROMEO:" --max-new 300
```

The checkpoint stores the model configuration, so no extra setup is needed. Pass --temperature and --top-k to control sampling.

## Architecture

- 6 layers, 6 heads, 384 embedding width
- context length 256 tokens
- vocabulary 1024 tokens
- GELU activation and LayerNorm before each block
- tied input and output embeddings
- {params:,} parameters in float32

## Training

- data: Tiny Shakespeare, 1,115,394 characters
- tokens: 399,354 train and 44,373 validation
- optimizer: AdamW with learning rate 0.001, warmup, and cosine decay
- batch: 32 sequences of 256 tokens
- planned steps: {planned}

The full training log and design notes live in the source repository.

## Notes

This repository uses a custom implementation, not the transformers library, so load it with the included code rather than AutoModelForCausalLM.

Data credit: Tiny Shakespeare from Andrej Karpathy's char rnn repository. The play text is public domain.
"""


def main():
    ap = argparse.ArgumentParser(description="package a checkpoint and upload it to hugging face")
    ap.add_argument("--repo-id", required=True, help="for example yourname/gpt_shakespeare")
    ap.add_argument("--ckpt", default="run/best.pt")
    ap.add_argument("--data-dir", default="data/shakespeare")
    ap.add_argument("--stage-dir", default="run/hf")
    ap.add_argument("--planned-steps", type=int, default=2500)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    tok = Path(args.data_dir) / "tok.json"
    for path in (ckpt, tok):
        if not path.exists():
            raise SystemExit(f"missing {path}; run prepare.py and train.py first")

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = GptConfig(**state["config"])
    params = Gpt(cfg).n_params
    card = CARD.format(
        params=params,
        vocab=cfg.vocab_size,
        step=state["step"],
        planned=args.planned_steps,
        loss=state["val_loss"],
        ppl=math.exp(state["val_loss"]),
    )

    stage = Path(args.stage_dir)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy(ckpt, stage / "best.pt")
    shutil.copy(tok, stage / "tok.json")
    for name in INFER_FILES:
        shutil.copy(name, stage / name)
    (stage / "README.md").write_text(card)

    files = sorted(stage.iterdir())
    print(f"staged {len(files)} files in {stage}")
    for path in files:
        print(f"  {path.name} {path.stat().st_size:,} bytes")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    api = HfApi(token=args.token)
    try:
        api.create_repo(args.repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(stage),
            repo_id=args.repo_id,
            commit_message="upload gpt checkpoint",
        )
    except Exception as err:
        raise SystemExit(
            f"upload failed: {err}\nlog in with `uv run hf auth login` or pass --token"
        )
    print(f"uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
