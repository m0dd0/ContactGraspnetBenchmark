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

        p_cam_x = ((img_points[:, 0] - cx) * img_points[:, 2] / fx).flatten()
        p_cam_y = ((img_points[:, 1] - cy) * img_points[:, 2] / fy).flatten()

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


# class IntrinsicsResizer:
#     def __init__(self, target_size):
#         self.target_size = target_size

#     def __call__(
#         self, K: NDArray[Shape["3, 3"], Float]
#     ) -> NDArray[Shape["3, 3"], Float]:
#         pass


# class SegmentationBoundingBoxer:
#     def __init__(self, margin: int = 0, seg_id: int = None):
#         self.margin = margin

#         self.seg_id = seg_id

#     def __call__(
#         self,
#         segmentation: NDArray[Shape["H, W"], Bool],
#         img: NDArray[Shape["H, W, 3"], Int] = None,
#     ) -> Tuple[NDArray[Shape["N,3"], Float]]:
#         if self.seg_id is not None:
#             segmentation = segmentation == self.seg_id

#         indices = np.argwhere(segmentation)

#         borders = np.array(
#             [
#                 max(0, indices[:, 0].min() - self.margin),
#                 min(img.shape[0], indices[:, 0].max() + self.margin),
#                 max(0, indices[:, 1].min() - self.margin),
#                 min(img.shape[1], indices[:, 1].max() + self.margin),
#             ]
#         )

#         img_box_segmented = img[borders[0] : borders[1], borders[2] : borders[3]]

#         return img_box_segmented


# class YCBSegmenter(ABC):
#     @abstractmethod
#     def __call__(
#         self, sample: YCBSimulationDataSample
#     ) -> Tuple[NDArray[Shape["N,3"], Float]]:
#         pass


# class YCBDataSegmenter(YCBSegmenter):
#     def __call__(
#         self, sample: YCBSimulationDataSample
#     ) -> Tuple[NDArray[Shape["N,3"], Float]]:
#         return sample.points_segmented, sample.points_segmented_color


# class YCBDepthBoxSegmenter(YCBSegmenter):
#     def __init__(self, depth2pc_converter: OrigDepth2Points = None, margin: int = 0):
#         self.depth2pc_converter = depth2pc_converter or OrigDepth2Points()

#         self.margin = margin

#     def _bounding_box(self, indices, img):
#         borders = np.array(
#             [
#                 max(0, indices[:, 0].min() - self.margin),
#                 min(img.shape[0], indices[:, 0].max() + self.margin),
#                 max(0, indices[:, 1].min() - self.margin),
#                 min(img.shape[1], indices[:, 1].max() + self.margin),
#             ]
#         )

#         img_box_segmented = img[borders[0] : borders[1], borders[2] : borders[3]]

#         return img_box_segmented

#     def __call__(
#         self, sample: YCBSimulationDataSample
#     ) -> Tuple[NDArray[Shape["N,3"], Float]]:
#         sample_pixels = np.argwhere(sample.segmentation)

#         depth_box_segmented = self._bounding_box(sample_pixels, sample.depth)
#         rgb_box_segmented = self._bounding_box(sample_pixels, sample.rgb)

#         points_box_segmented, points_color_box_segmented = self.depth2pc_converter(
#             depth_box_segmented, sample.cam_intrinsics, rgb_box_segmented
#         )

#         return points_box_segmented, points_color_box_segmented


# class OrigDepth2Points:
#     def __call__(
#         self,
#         depth: NDArray[Shape["H, W"], Float],
#         K: NDArray[Shape["3, 3"], Float],
#         rgb: NDArray[Shape["H, W, 3"], Int] = None,
#     ) -> Tuple[NDArray[Shape["N,3"], Float]]:
#         mask = np.where(depth > 0)
#         x, y = mask[1], mask[0]  # x indices, y indices

#         normalized_x = x.astype(np.float32) - K[0, 2]
#         normalized_y = y.astype(np.float32) - K[1, 2]

#         world_x = normalized_x * depth[y, x] / K[0, 0]
#         world_y = normalized_y * depth[y, x] / K[1, 1]
#         world_z = depth[y, x]

#         pc = np.vstack((world_x, world_y, world_z)).T

#         if rgb is not None:
#             rgb = rgb[y, x, :]

#         return (pc, rgb)


# class SegmenterPixel:
#     def __init__(self, seg_id: int = None):
#         self.seg_id = seg_id

#     def __call__(
#         self,
#         segmentation: NDArray[Shape["H, W"], Bool],
#         depth_img: NDArray[Shape["H, W"], Int],
#         rgb_img: NDArray[Shape["H, W, 3"], Int] = None,
#     ) -> Tuple[NDArray[Shape["N,3"], Float], NDArray[Shape["N,3"], Int]]:
#         if self.seg_id is not None:
#             segmentation = segmentation == self.seg_id

#         indices = np.argwhere(segmentation)

#         img_coords = np.vstack(
#             (indices[:, 0], indices[:, 1], depth_img[indices[:, 0], indices[:, 1]])
#         ).T

#         colors = None
#         if rgb_img is not None:
#             colors = rgb_img[indices[:, 0], indices[:, 1]]

#         return img_coords, colors
