"""This module contains the custom transforms that are used in the postprocessing pipelines.
They should be as concise as possible and only contain the logic that is necessary to
execute a singe transformation step.
They might also be used directly in a Compose to make a descriptive pipeline.
"""

# import torch
# from torchvision import transforms as T
from torchtyping import TensorType

class ExampleSubmodule:
    def __init__(self):
        super().__init__()

        # configuration flags here

    def __call__(self, x: TensorType) -> TensorType:
        # TODO implement submodule
        raise NotImplementedError()
        return x

