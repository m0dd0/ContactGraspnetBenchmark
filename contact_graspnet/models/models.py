""" This module contains the network architecture(s) of the project."""

from torchtyping import TensorType

# from .custom_modules import CustomModuleExample
from .base import BaseModel

class ContactGraspnet(BaseModel):
    def __init__(self):
        super().__init__()



    def forward(self, x: TensorType) -> TensorType:
        # TODO: implement the forward pass of the network here
        
        raise NotImplementedError()
    