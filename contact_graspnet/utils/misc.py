from typing import Tuple

import numpy as np
from nptyping import NDArray, Shape, Int, Float


def posrot2pose(
    pos: NDArray[Shape["3"], Float], rot: NDArray[Shape["3,3"], Float]
) -> NDArray[Shape["4,4"], Float]:
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def pose2posrot(
    pose: NDArray[Shape["4,4"], Float]
) -> Tuple[NDArray[Shape["3"], Float], NDArray[Shape["3,3"], Float]]:
    return pose[:3, 3], pose[:3, :3]
