"""This module contains submodules of the network architecture(s) like special layers etc."""

import torch.nn as nn

from torchtyping import TensorType


class CustomModuleExample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorType) -> TensorType:
        raise NotImplementedError()
