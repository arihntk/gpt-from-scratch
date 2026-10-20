from pathlib import Path

import numpy as np
import torch


class Loader:
    def __init__(self, data_dir, split):
        path = Path(data_dir) / f"{split}.bin"
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n = len(self.data)

    def sample(self, batch, block, device):
        ix = torch.randint(self.n - block - 1, (batch,))
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + block].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(self.data[i + 1 : i + 1 + block].astype(np.int64)) for i in ix]
        )
        return x.to(device), y.to(device)
