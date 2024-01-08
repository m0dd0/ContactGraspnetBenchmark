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
from typing import Any, Dict, Union, Tuple

from nptyping import NDArray, Shape, Float
import numpy as np

from contact_graspnet.datatypes import (
    DatasetSample,
    YCBSimulationDataSample,
    OrigExampleDataSample,
)
from . import custom_transforms as CT


class PreprocessorBase(ABC):
    def __init__(self):
        super().__init__()
        self.intermediate_results: Dict[str, Any] = {}

    @abstractmethod
    def __call__(self, sample: DatasetSample) -> NDArray[Shape["N,3"], Float]:
        pass


class UniversalPreprocessor(PreprocessorBase):
    def __init__(
        self,
        depth2points_converter: CT.Depth2ImgPoints,
        img2cam_converter: CT.Img2CamCoords,
        z_clipper: CT.ZClipper,
        segmentation_binarizer: CT.SegmentationBinarizer = None,
        resizer: CT.Resizer = None,
    ):
        super().__init__()

        self.depth2points_converter = depth2points_converter
        self.img2cam_converter = img2cam_converter
        self.z_clipper = z_clipper
        self.segmentation_binarizer = segmentation_binarizer or (lambda x: x)
        self.resizer = resizer or (lambda x, y=None: (x, y))

    def __call__(
        self,
        sample: Union[OrigExampleDataSample, YCBSimulationDataSample],
    ) -> Tuple[NDArray[Shape["N,3"], Float], NDArray[Shape["M,3"], Float]]:
        depth, intrinsics = self.resizer(sample.depth, sample.cam_intrinsics)
        rgb, _ = self.resizer(sample.rgb)
        segmentation, _ = self.resizer(sample.segmentation)

        segmentation = self.segmentation_binarizer(segmentation)

        full_pc, full_pc_colors = self.depth2points_converter(depth, rgb)
        full_pc = self.img2cam_converter(full_pc, intrinsics)
        full_pc, full_pc_colors = self.z_clipper(full_pc, full_pc_colors)

        segmented_pc, segmented_pc_colors = self.depth2points_converter(
            depth, rgb, segmentation
        )
        segmented_pc = self.img2cam_converter(segmented_pc, intrinsics)
        segmented_pc, segmented_pc_colors = self.z_clipper(
            segmented_pc, segmented_pc_colors
        )

        self.intermediate_results["full_pc_colors"] = full_pc_colors
        self.intermediate_results["segmented_pc_colors"] = segmented_pc_colors
        self.intermediate_results["depth"] = depth
        self.intermediate_results["rgb"] = rgb
        self.intermediate_results["segmentation"] = segmentation
        self.intermediate_results["initial_sample"] = sample

        return full_pc, segmented_pc