"""This module contains utilities to visualize the different data created during processing.
In contrast to the other parts of this project the visualization utitlities are not
designed to be used as a pipeline/processor. Instead functions are provided that can be
used to visualize the data in a notebook or in a script.
Try to use functions which operate on matplotlib.Axes objects as this makes it easier
to combine the different visualizations.
"""

from pathlib import Path
from typing import Tuple, List

from matplotlib import pyplot as plt
import numpy as np
from nptyping import NDArray, Shape, Int, Float
import mayavi.mlab as mlab

# from contact_graspnet.utils.misc import posrot2pose
from contact_graspnet.orig.contact_graspnet import visualization_utils as orig_vis
from contact_graspnet.datatypes import GraspPaperCam, GraspWorld
from contact_graspnet.postprocessing import World2ImgCoordConverter


def make_tensor_displayable(
    tensor, convert_chw: bool = False, convert_to_int: bool = False
):
    """Executes different operations to make a tensor displayable.
    The tensor is always converted to a numpy array and squeezed.

    Args:
        tensor: A datastrucutre that can be converted to a numpy array.
        convert_chw (bool, optional): Converter a chw tensor to a hwc tensor.
            Defaults to False.
        convert_to_int (bool, optional): Converts the datatype to uint8. Defaults to False.

    Returns:
        _type_: _description_
    """
    tensor = np.array(tensor)
    tensor = np.squeeze(tensor)  # convert one channel images to 2d tensors
    assert len(tensor.shape) in [2, 3], "squeezed Tensor must be 2 or 3 dimensional"

    if convert_chw:
        assert tensor.shape[0] == 3, "first dimension must be 3"
        # chw -> hwc
        tensor = np.transpose(tensor, (1, 2, 0))

    if convert_to_int:
        tensor = tensor.astype("uint8")

    return tensor


def visualize_pointcloud(
    ax, pointcloud, pointcloud_colors=None, max_points=None, size=None
):
    """Visualizes a pointcloud.

    Args:
        ax (matplotlib.Axes): The axes to plot on.
        pointcloud (np.ndarray): The pointcloud to visualize.
        pointcloud_colors (np.ndarray, optional): The colors of the points.
            Defaults to None.
        max_points (int, optional): The maximum number of points to visualize.
            Defaults to None.
    """
    if max_points is not None and pointcloud.shape[0] > max_points:
        nth = pointcloud.shape[0] // max_points
        pointcloud = pointcloud[::nth]
        if pointcloud_colors is not None:
            pointcloud_colors = pointcloud_colors[::nth]

    ax.scatter(
        pointcloud[:, 0],
        pointcloud[:, 1],
        pointcloud[:, 2],
        c=pointcloud_colors,
        s=size or max(0.1, pointcloud.shape[0] / 1000),
        marker=".",
    )


def mlab_pose_vis(
    pointcloud: NDArray[Shape["N, 3"], Float],
    grasps: List[GraspPaperCam],
    pointcloud_colors: NDArray[Shape["N, 3"], Float] = None,
    image_path: Path = None,
    image_size: Tuple = (640, 480),
):
    orig_vis.visualize_grasps(
        full_pc=pointcloud,
        pred_grasps_cam={-1: np.array([g.pose for g in grasps])},
        scores={-1: np.array([g.score for g in grasps])},
        plot_opencv_cam=False,
        pc_colors=pointcloud_colors,
        gripper_openings={-1: np.array([g.width for g in grasps])},
        gripper_width=None,
    )

    if image_path is None:
        mlab.show()
    else:
        mlab.savefig(str(image_path), size=image_size)
        mlab.close()


def world_grasps_ax(
    ax,
    orig_rgb,
    grasps: List[GraspWorld],
    cam_intrinsics,
    cam_rot,
    cam_pos,
    annotate: bool = True,
):
    ax.imshow(orig_rgb)

    world2img_converter = World2ImgCoordConverter()

    for grasp in grasps:
        center_img = world2img_converter(
            grasp.position, cam_intrinsics, cam_rot, cam_pos
        )
        ax.scatter(x=center_img[0], y=center_img[1])

        # antipodal_points_world = get_antipodal_points(
        #     grasp.center[0:2], grasp.angle, grasp.width
        # )
        # antipodal_points_world = np.hstack(
        #     (antipodal_points_world, np.full((2, 1), grasp.center[2]))
        # )

        # antipodal_points_img = np.array(
        #     [
        #         world2img_converter(p, cam_intrinsics, cam_rot, cam_pos)
        #         for p in antipodal_points_world
        #     ]
        # )
        # ax.plot(antipodal_points_img[:, 0], antipodal_points_img[:, 1])

        if annotate:
            ax.annotate(
                f"c: {tuple(grasp.position.round(3))}\n"
                + f"q: {round(grasp.score, 3)}\n"
                # + f"a: {round(np.rad2deg(grasp.angle), 3)}\n"
                + f"w: {round(grasp.width, 3)}",
                xy=center_img[0:2],
            )


def overview_fig(
    # different data to visualize
    fig=None,
):
    if fig is None:
        fig = plt.figure()

    # TODO add different data to visualize

    raise NotImplementedError()

    return fig
