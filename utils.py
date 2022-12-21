import typing as tp

import torch

from torch.nn.utils.rnn import pad_sequence

def collator(
    batch: tp.List[torch.Tensor]
) -> torch.Tensor:
    return pad_sequence(batch, batch_first=True)
