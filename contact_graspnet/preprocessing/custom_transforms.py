"""This module contains the custom transforms that are used in the preprocessing pipelines.
They should be as concise as possible and only contain the logic that is necessary to
execute a singe transformation step.
They might also be used directly in a Compose to make a descriptive pipeline.
"""

from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np
from nptyping import NDArray, Shape, Float, Int

from contact_graspnet.datatypes import YCBSimulationDataSample


class OrigDepth2Points:
    def __call__(
        self,
        depth: NDArray[Shape["H, W"], Float],
        K: NDArray[Shape["3, 3"], Float],
        rgb: NDArray[Shape["H, W, 3"], Int] = None,
    ) -> Tuple[NDArray[Shape["N,3"], Float]]:
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
        self,
        pointcloud: NDArray[Shape["N,3"], Float],
        pointcloud_colors: NDArray[Shape["N,3"], Int] = None,
    ) -> Tuple[NDArray[Shape["N,3"], Float]]:
        mask = np.logical_and(
            pointcloud[:, 2] > self.z_range[0], pointcloud[:, 2] < self.z_range[1]
        )

        pointcloud_filtered = pointcloud[mask]
        pointcloud_colors_filtered = (
            pointcloud_colors[mask] if pointcloud_colors is not None else None
        )

        return pointcloud_filtered, pointcloud_colors_filtered


class YCBSegmenter(ABC):
    @abstractmethod
    def __call__(
        self, sample: YCBSimulationDataSample
    ) -> Tuple[NDArray[Shape["N,3"], Float]]:
        pass


class YCBDataSegmenter(YCBSegmenter):
    def __call__(
        self, sample: YCBSimulationDataSample
    ) -> Tuple[NDArray[Shape["N,3"], Float]]:
        return sample.points_segmented, sample.points_segmented_color


class YCBDepthBoxSegmenter(YCBSegmenter):
    def __init__(self, depth2pc_converter: OrigDepth2Points = None, margin: int = 0):
        self.depth2pc_converter = depth2pc_converter or OrigDepth2Points()

        self.margin = margin

    def _bounding_box(self, indices, img):
        borders = np.array(
            [
                max(0, indices[:, 0].min() - self.margin),
                min(img.shape[0], indices[:, 0].max() + self.margin),
                max(0, indices[:, 1].min() - self.margin),
                min(img.shape[1], indices[:, 1].max() + self.margin),
            ]
        )

        img_box_segmented = img[borders[0] : borders[1], borders[2] : borders[3]]

        return img_box_segmented

    def __call__(
        self, sample: YCBSimulationDataSample
    ) -> Tuple[NDArray[Shape["N,3"], Float]]:
        sample_pixels = np.argwhere(sample.segmentation)

        depth_box_segmented = self._bounding_box(sample_pixels, sample.depth)
        rgb_box_segmented = self._bounding_box(sample_pixels, sample.rgb)

        points_box_segmented, points_color_box_segmented = self.depth2pc_converter(
            depth_box_segmented, sample.cam_intrinsics, rgb_box_segmented
        )

        return points_box_segmented, points_color_box_segmented
