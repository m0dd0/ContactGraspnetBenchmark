from dataclasses import dataclass

from nptyping import NDArray, Float, Shape

# here we define the datatypes we use in the project
# theses should contain only data and no logic as we want to use the dataloaders and pipelines
# for conversions etc
# so classmethods in the style of .from_xy() etc should be avoided


@dataclass
class CameraData:
    rgb: NDArray[Shape["3,h,w"], Float]
    depth: NDArray[Shape["1,h,w"], Float]
    points: NDArray[Shape["n,3"], Float]
    segmentation: NDArray[Shape["1,h,w"], Float]
    name: str
    pos_grasps: NDArray[Shape["n_pos_grasps,4,2"], Float] = None
    neg_grasps: NDArray[Shape["n_pos_grasps,4,2"], Float] = None
    cam_intrinsics: NDArray[Shape["3, 3"], Float] = None
    cam_pos: NDArray[Shape["3"], Float] = None
    cam_rot: NDArray[Shape["4"], Float] = None
