""" This module contains the network architecture(s) of the project."""

import torch.nn as nn

from torchtyping import TensorType

from .custom_modules import CustomModuleExample
from .base import BaseModel

class ContactGraspnet(BaseModel):
    def __init__(
            self, 
            # model_param_1, 
            # model_param_2
        ):
        super().__init__()

        # TODO: configure the modules of the network here
        # self.model_param_1 = model_param_1
        # self.model_param_2 = model_param_2

        raise NotImplementedError()

    def forward(self, x: TensorType) -> TensorType:
        # TODO: implement the forward pass of the network here
        
        raise NotImplementedError()
    