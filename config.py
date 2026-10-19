from dataclasses import asdict, dataclass


@dataclass
class GptConfig:
    vocab_size: int = 1024
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

    def dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 32
    steps: int = 3000
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup: int = 100
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    eval_interval: int = 50
    eval_iters: int = 20
    seed: int = 1337


DEFAULT_MODEL = GptConfig()
DEFAULT_TRAIN = TrainConfig()
