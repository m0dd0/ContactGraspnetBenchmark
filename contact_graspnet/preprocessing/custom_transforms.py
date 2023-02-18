"""This module contains the custom transforms that are used in the preprocessing pipelines.
They should be as concise as possible and only contain the logic that is necessary to
execute a singe transformation step.
They might also be used directly in a Compose to make a descriptive pipeline.
"""

import numpy as np
from nptyping import NDArray, Shape, Float, Int


class OrigDepth2Points:
    def __call__(
        self,
        depth: NDArray[Shape["H, W"], Float],
        K: NDArray[Shape["3, 3"], Float],
        rgb: NDArray[Shape["H, W, 3"], Int] = None,
    ) -> NDArray[Shape["N,3"], Float]:
        mask = np.where(depth > 0)
        x, y = mask[1], mask[0]

        normalized_x = x.astype(np.float32) - K[0, 2]
        normalized_y = y.astype(np.float32) - K[1, 2]

        world_x = normalized_x * depth[y, x] / K[0, 0]
        world_y = normalized_y * depth[y, x] / K[1, 1]
        world_z = depth[y, x]

        pc = np.vstack((world_x, world_y, world_z)).T

        if rgb is not None:
            rgb = rgb[y, x, :]

        return (pc, rgb)


class ZClipper:
    def __init__(self, z_range: NDArray[Shape["2"], Float]):
        self.z_range = z_range

    def __call__(
        self, pointcloud: NDArray[Shape["N,3"], Float]
    ) -> NDArray[Shape["N,3"], Float]:
        mask = np.logical_and(
            pointcloud[:, 2] > self.z_range[0], pointcloud[:, 2] < self.z_range[1]
        )

        return pointcloud[mask]
