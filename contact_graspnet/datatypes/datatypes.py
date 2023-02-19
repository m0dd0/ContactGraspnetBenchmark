"""This module contains only dataclasses. These dataclasses hold only information and no
logic. They are used to define the datatypes used in the project.
The datatypes can be grouped into two categories:
- DatasetSamples: These are the samples that are returned by the datasets
There should be an compatible preprocessing pipeline which accepts these samples as input
in order to used them in the project.
- Results: These are the results that are returned by the posprocessing steps
There should be an compatible postprocessing pipeline which outputs these dataclasses as results.
"""
from dataclasses import dataclass
from abc import ABC

from nptyping import NDArray, Shape, Float, Int


@dataclass
class DatasetSample(ABC):
    name: str


@dataclass
class YCBSimulationDataSample(DatasetSample):
    name: str
    rgb: NDArray[Shape["H, W, 3"], Int]
    depth: NDArray[Shape["H, W"], Float]
    segmentation: NDArray[Shape["H, W"], Int]
    points: NDArray[Shape["N, 3"], Float]
    points_color: NDArray[Shape["N, 3"], Int]
    points_segmented: NDArray[Shape["N, 3"], Float]
    cam_intrinsics: NDArray[Shape["3, 3"], Float]
    cam_pos: NDArray[Shape["3"], Float]
    cam_rot: NDArray[Shape["3, 3"], Float]


@dataclass
class OrigExampleDataSample(DatasetSample):
    name: str
    segmentation: NDArray[Shape["H, W"], Int]  # (720,1280) 0 - 5
    rgb: NDArray[Shape["H, W, 3"], Int]  # (720,1280,3) 0-255
    depth: NDArray[Shape["H, W"], Float]  # (720,1280) 0 - 1
    cam_intrinsics: NDArray[Shape["3, 3"], Float]  # (3,3)


@dataclass
class ResultBase(ABC):
    pass


@dataclass
class GraspCam(ResultBase):
    score: float
    pos: NDArray[Shape["3"], Float]
    orientation: NDArray[Shape["3, 3"], Float]
    contact_point: NDArray[Shape["3"], Float]
    width: float
