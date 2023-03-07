"""This module contains the custom transforms that are used in the preprocessing pipelines.
They should be as concise as possible and only contain the logic that is necessary to
execute a singe transformation step.
They might also be used directly in a Compose to make a descriptive pipeline.
"""

from typing import Tuple, Union
from copy import deepcopy

import numpy as np
from nptyping import NDArray, Shape, Float, Int
import skimage as ski

from contact_graspnet.datatypes import YCBSimulationDataSample, OrigExampleDataSample


class SegmentationBinarizer:
    def __init__(self, segmentation_id: int):
        self.segmentation_id = segmentation_id

    def __call__(
        self, segmentation: NDArray[Shape["H, W"], Int]
    ) -> NDArray[Shape["H, W"], Int]:
        return segmentation == self.segmentation_id


class Depth2ImgPoints:
    def __call__(
        self,
        depth_img: NDArray[Shape["H, W"], Float],
        rgb_img: NDArray[Shape["H, W, 3"], Float] = None,
        segmentation: NDArray[Shape["H, W"], Int] = None,
    ) -> Tuple[NDArray[Shape["N,3"], Float], NDArray[Shape["N,3"], Float]]:
        if segmentation is None:
            segmentation = np.ones_like(depth_img, dtype=bool)

        indices = np.argwhere(segmentation)

        img_coords = np.vstack(
            (indices[:, 0], indices[:, 1], depth_img[indices[:, 0], indices[:, 1]])
        ).T

        colors = None
        if rgb_img is not None:
            colors = rgb_img[indices[:, 0], indices[:, 1]]

        return img_coords, colors


class Img2CamCoords:
    # def __init__(self):
    #     pass

    def __call__(
        self,
        img_points: NDArray[Shape["N,3"], Float],
        K: NDArray[Shape["3, 3"], Float],
    ) -> NDArray[Shape["N,3"], Float]:
        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        p_cam_x = ((img_points[:, 1] - cx) * img_points[:, 2] / fx).flatten()
        p_cam_y = ((img_points[:, 0] - cy) * img_points[:, 2] / fy).flatten()

        cam_points = np.vstack((p_cam_x, p_cam_y, img_points[:, 2])).T

        return cam_points


class ZClipper:
    def __init__(self, z_range: NDArray[Shape["2"], Float]):
        self.z_range = z_range

    def __call__(
        self,
        pointcloud: NDArray[Shape["N,3"], Float],
        pointcloud_colors: NDArray[Shape["N,3"], Int] = None,
    ) -> Tuple[NDArray[Shape["N,3"], Float], NDArray[Shape["N,3"], Int]]:
        mask = np.logical_and(
            pointcloud[:, 2] > self.z_range[0], pointcloud[:, 2] < self.z_range[1]
        )

        pointcloud_filtered = pointcloud[mask]
        pointcloud_colors_filtered = (
            pointcloud_colors[mask] if pointcloud_colors is not None else None
        )

        return pointcloud_filtered, pointcloud_colors_filtered


class Resizer:
    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size

    def __call__(
        self,
        img: Union[NDArray[Shape["H, W, 3"], Float], NDArray[Shape["H, W"], Float]],
        K: NDArray[Shape["3, 3"], Float] = None,
    ) -> Tuple[
        Union[NDArray[Shape["H, W, 3"], Float], NDArray[Shape["H, W"], Float]],
        NDArray[Shape["3, 3"], Float],
    ]:
        if K is not None:
            assert img.shape[1] == K[0, 2] * 2
            assert img.shape[0] == K[1, 2] * 2

            K = K.copy()
            factor_x = (self.target_size[0] / 2) / K[1, 2]
            factor_y = (self.target_size[1] / 2) / K[0, 2]

            K[0, :] *= factor_x
            K[1, :] *= factor_y

        return ski.transform.resize(img, self.target_size), K