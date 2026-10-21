# gpt

A small GPT2 style language model implemented from scratch in PyTorch. Everything is in this repository: the byte level BPE tokenizer, the transformer, the training loop, and the sampling scripts. No pretrained weights, no tokenizer downloads, and no model frameworks beyond PyTorch and NumPy.

Training status: not complete. The run stopped at step 60 of the planned 2500 steps. All training numbers and samples below come from that partial run. The full log lives in TRAIN.md.

## Highlights

* 11,139,072 parameters in a 6 layer GPT2 style decoder
* byte level BPE tokenizer with a 1024 token vocabulary trained on the corpus
* trains on a MacBook Air M4 with 16 GB memory in about 22 minutes for the full plan
* complete pipeline: prepare, train, sample, and publish to Hugging Face

## Quick start

Requires Python 3.11 or newer and uv.

```
uv sync
uv run python prepare.py
uv run python train.py
uv run python generate.py --prompt "ROMEO:"
```

prepare.py downloads Tiny Shakespeare, trains the tokenizer, and writes the train and validation files. train.py follows the default plan of 2500 steps.

## Model

* parameters: 11,139,072
* layers: 6
* attention heads: 6
* embedding width: 384
* context length: 256 tokens
* vocabulary: 1024 tokens
* activation: GELU with the tanh approximation
* normalization: LayerNorm before each attention block and each MLP block
* dropout: 0.1
* output head: tied to the token embeddings
* precision: float32

The architecture follows GPT2: learned position embeddings, causal self attention, MLP blocks with a 4x expansion, and a final LayerNorm before the language model head. Weight init uses a normal distribution with standard deviation 0.02, and the projection matrices in each residual branch use a smaller scale of 0.02 divided by the square root of twice the layer count.

Parameter breakdown:

* token embeddings: 393,216
* position embeddings: 98,304
* 6 transformer blocks: 10,646,784
* final LayerNorm: 768
* total: 11,139,072

The output head shares its weights with the token embeddings, so the unique parameter count is 11,139,072. A float32 checkpoint is about 45 MB.

## Data

* corpus: Tiny Shakespeare, public domain play text from Andrej Karpathy's char rnn repository
* size: 1,115,394 characters and bytes
* tokenizer: 768 merges and 1024 vocabulary entries
* tokens after tokenization: 443,727
* compression: 2.51 bytes per token
* train split: 399,354 tokens
* validation split: 44,373 tokens
* split rule: 90/10 by position

## Hardware

* machine: MacBook Air M4 with 16 GB unified memory
* system: macOS on arm64
* Python: 3.14
* PyTorch: 2.14.1
* NumPy: 2.5.4
* compute backend: MPS, Apple Metal
* memory use: well under 1 GB for the model, optimizer, and activations

## Training plan

* optimizer: AdamW with beta values 0.9 and 0.95
* learning rate: 0.001 with 100 step warmup and cosine decay to 10 percent of the peak
* weight decay: 0.1 on matrices only
* gradient clipping: norm 1.0
* batch: 32 sequences of 256 tokens, so 8192 tokens per step
* steps: 2500, about 51 passes over the training data
* evaluation: every 50 steps over 20 batches per split
* sampling: at step 0 and at the final step, temperature 0.8 and top k 64
* checkpoints: best validation loss and last state
* seed: 1337

## Training time

Measured on the M4 Air:

* steady state step time: about 0.43 s
* throughput: about 19,000 tokens per second
* first step: 1 to 3 s while Metal kernels compile
* evaluation cost: about 4 to 5 s every 50 steps
* projected full run: about 18 minutes of training and about 4 minutes of evaluation, so about 22 minutes total

The partial run reached step 60 in 37 s of wall clock time, including the first evaluation and the first sample.

## Loss curve so far

Training is not complete, so this is the start of the curve only.

Batch loss at the logged steps:

* step 0: 7.0186
* step 10: 6.6099
* step 20: 6.3556
* step 30: 6.1517
* step 40: 5.8264
* step 50: 5.3838
* step 60: 5.0601

Evaluation loss:

* step 0: train 6.9762, validation 6.9726, validation perplexity 1067.02
* step 50: train 5.3354, validation 5.4735, validation perplexity 238.29

For reference, a uniform guess over 1024 tokens gives loss ln(1024) = 6.93. The model passed that level quickly, which shows that the tokenizer and the training loop work as expected.

## Samples

Untrained model at step 0:

```
see thatthinkIfeelifs repchloogood Is Is et

fore cessI thoumZmarlorWhy, she timothst oll, my see INGas I ...
```

Step 50 checkpoint, temperature 0.8, top k 64, 100 new tokens:

```
thth: thrones:
ney a bort thou sil,
I opven cnear
Thon:
Drow
Thing Sha ring the band hour train his his s; ford weep tha mudeed Bent you wit
```

The model has learned word shapes, line breaks, and some character names, but not grammar or meaning. That is expected after 2 percent of the planned training.

## Hugging Face upload

upload.py packages a checkpoint, the tokenizer, and the inference code, generates a model card, and pushes the result to the Hugging Face Hub.

Log in once:

```
uv run hf auth login
```

Upload a checkpoint:

```
uv run python upload.py --repo-id yourname/gpt_shakespeare
```

Options:

* `--ckpt`: path to the checkpoint, default run/best.pt
* `--data-dir`: directory that contains tok.json, default data/shakespeare
* `--stage-dir`: local folder for the upload package, default run/hf
* `--repo-id`: full repository name, required
* `--private`: creates a private repository
* `--token`: access token, otherwise the saved login or HF_TOKEN is used
* `--dry-run`: builds the package without uploading
* `--planned-steps`: step count shown on the model card, default 2500

Uploaded files:

* best.pt
* tok.json
* config.py
* model.py
* tokenizer.py
* generate.py
* README.md, generated model card

The model card shows the parameter count, architecture, training status, checkpoint metrics, and usage commands. The repository is a custom implementation and not a transformers model, so it must be loaded with the included code and not with AutoModelForCausalLM.

## Files

* config.py: model and training settings, device selection
* tokenizer.py: byte level BPE tokenizer with fit, encode, decode, save, and load
* model.py: attention, MLP, transformer block, GPT model, and sampling
* dataset.py: memory mapped train and validation batches
* prepare.py: downloads the corpus, trains the tokenizer, writes token files
* train.py: training loop with warmup, cosine decay, evaluation, sampling, checkpoints, and resume
* generate.py: loads a checkpoint and samples text
* upload.py: packages a checkpoint and publishes it to Hugging Face
* TRAIN.md: local training log, not tracked by git
* DECISIONS.md: local design notes, not tracked by git

## Limitations

* Training is not complete. The model produces word shaped noise, not usable text.
* The corpus is about 1.1 million characters, so the model can only learn the style of one small collection of plays.
* The validation split has 44,373 tokens, so validation loss moves by a few hundredths between evaluations.
* An interrupted run cannot resume unless last.pt was written at the end of an earlier run.
* The implementation favors clarity over speed. There is no torch.compile, mixed precision, or distributed training.
* Checkpoints are not part of the git repository. They are written to run/ and ignored, so a fresh clone needs its own training run.

## Credits

Data: Tiny Shakespeare from Andrej Karpathy's char rnn repository. The play text is public domain.
Architecture ideas follow GPT2 and the nanoGPT teaching project.
