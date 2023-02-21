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
    ):
        super().__init__()

        self.depth2points_converter = depth2points_converter
        self.img2cam_converter = img2cam_converter
        self.z_clipper = z_clipper

    def __call__(
        self,
        sample: Union[OrigExampleDataSample, YCBSimulationDataSample],
    ) -> Tuple[NDArray[Shape["N,3"], Float], NDArray[Shape["M,3"], Float]]:
        assert (
            len(np.unique(sample.segmentation)) == 2
        ), "Segmentation should only contain two classes or segmentation id needs to be specified."

        full_pc, full_pc_colors = self.depth2points_converter(sample.depth, sample.rgb)
        full_pc = self.img2cam_converter(full_pc, sample.cam_intrinsics)
        full_pc, full_pc_colors = self.z_clipper(full_pc, full_pc_colors)

        segmented_pc, segmented_pc_colors = self.depth2points_converter(
            sample.depth, sample.rgb, sample.segmentation
        )
        segmented_pc = self.img2cam_converter(segmented_pc, sample.cam_intrinsics)
        segmented_pc, segmented_pc_colors = self.z_clipper(
            segmented_pc, segmented_pc_colors
        )

        self.intermediate_results["full_pc_colors"] = full_pc_colors
        self.intermediate_results["segmented_pc_colors"] = segmented_pc_colors

        return full_pc, segmented_pc


# class YCBSimulationPreprocessor(PreprocessorBase):
#     def __init__(self, z_clipper: CT.ZClipper, segmenter: CT.YCBSegmenter = None):
#         super().__init__()

#         self.z_clipper = z_clipper
#         self.segmenter = segmenter

#     def __call__(self, sample: YCBSimulationDataSample) -> NDArray[Shape["N,3"], Float]:
#         points = sample.points
#         points_color = sample.points_color

#         if self.segmenter is not None:
#             points, points_color = self.segmenter(sample)

#         if self.z_clipper is not None:
#             points, points_color = self.z_clipper(points, points_color)

#         self.intermediate_results["pointcloud_colors"] = points_color

#         return points


# other preprocessors for other datasets or with completely different preprocessing pipelines ...
