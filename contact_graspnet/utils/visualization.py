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
import matplotlib as mpl

# from contact_graspnet.utils.misc import posrot2pose
from contact_graspnet.orig.contact_graspnet import visualization_utils as orig_vis
from contact_graspnet.datatypes import GraspPaperCam, GraspWorld
from contact_graspnet.postprocessing import World2ImgCoordConverter


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


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


def world_grasps_center_ax(
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


def grasp6D_ax(
    ax: mpl.axes.Axes,
    grasp: GraspWorld,
    pc_world: NDArray[Shape["N, 3"], Float],
    pc_colors: NDArray[Shape["N, 3"], Float] = None,
    pointsize: float = 0.1,
):
    ax.scatter(
        pc_world[:, 0], pc_world[:, 1], pc_world[:, 2], c=pc_colors / 255, s=pointsize
    )

    a = grasp.orientation[:, 2].flatten() * 0.5
    a_axis_points = np.array([grasp.position, grasp.position - a]).T
    ax.plot(
        xs=a_axis_points[0],
        ys=a_axis_points[1],
        zs=a_axis_points[2],
        zorder=1,
    )

    b = grasp.orientation[:, 0].flatten() * 0.5
    b_axis_points = np.array([grasp.position + 0.5 * b, grasp.position - 0.5 * b]).T
    ax.plot(
        xs=b_axis_points[0],
        ys=b_axis_points[1],
        zs=b_axis_points[2],
        zorder=1,
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
