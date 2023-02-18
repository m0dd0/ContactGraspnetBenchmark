""" This modules contains the preprocessing pipelines. A preprocessing pipeline should
accept a sample from a dataset and return a tensor that can be used as input for the
network. The preprocessing pipeline should also store intermediate results in the
intermediate_results dictionary. These results can be used in the end for closer evauation
and debugging.
Since a pipeline is not used in backpropagation it is not necessary to implement it as
a torch.nn.Module.
A pipeline should consist out of multiple submodules that are called in a specific order.
For clarity it should be avoided to have logic inthe pipeline itself. Instead the logic
should be implemented in the submodules. The pipeline itself should only manage the
flow of information through the submodules. If a pipeline has no or only a single
submodule it might be more suitable to implement it as a submodule instead of a pipeline
for improved reusability. It is also good practice to have the pipeline accept (multiple)
submodules as arguments. This way the pipeline can be used with different submodules
which increases modularity.
Somteimes it might be useful to have a subpipeline that is used in multiple pipelines.
This subpipeline will output an intermediate result that is used in multiple pipelines
but not the final result. Keep in mind that you need to manage the intermediate result 
of the subpipeline yourself.
"""

from abc import abstractmethod, ABC
from typing import Any, Dict

# from torchvision import transforms as T
# import torch
# import numpy as np
from torchtyping import TensorType

from contact_graspnet.datatypes import DatasetSample, YCBSimulationDataSample
from . import custom_transforms as CT


class PreprocessorBase(ABC):
    def __init__(self):
        super().__init__()
        self.intermediate_results: Dict[str, Any] = {}

    @abstractmethod
    def __call__(self, sample: DatasetSample) -> TensorType["batch", ...]:
        pass


class YCBSimulationDataPreprocessor(PreprocessorBase):
    def __init__(
        self,
        # submodule_1: CT.ExampleSubmodule = None,
        # submodule_2: CT.ExampleSubmodule = None
    ):
        super().__init__()

        # submodules
        # TODO add submodules here
        # self.example_submodule_1 = exmaple_submodule_1
        # self.example_submodule_2 = exmaple_submodule_2

    def __call__(self, sample: YCBSimulationDataSample) -> TensorType["batch", ...]:
        # TODO implement preprocessing pipeline
        # x = network_input
        # if self.example_submodule_1 is not None:
        #     x = self.example_submodule_2(x)
        # if self.example_submodule_2 is not None:
        #     x = self.example_submodule_2(x)
        # return x
        raise NotImplementedError()


# other preprocessors for other datasets or with completely different preprocessing pipelines ...
