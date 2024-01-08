"""This module contains the custom transforms that are used in the postprocessing pipelines.
They should be as concise as possible and only contain the logic that is necessary to
execute a singe transformation step.
They might also be used directly in a Compose to make a descriptive pipeline.
"""
from typing import List

from nptyping import NDArray, Shape, Float
import numpy as np

from contact_graspnet.datatypes import GraspCam, GraspPaperCam


class TopScoreFilter:
    def __init__(self, top_k: int = 1):
        self.top_k = top_k

    def __call__(self, grasps: List[GraspCam]) -> List[GraspCam]:
        if len(grasps) <= self.top_k:
            return grasps

        return sorted(grasps, key=lambda grasp: grasp.score, reverse=True)[: self.top_k]


class Cam2WorldCoordConverter:
    def __init__(self):
        pass

    def __call__(
        self,
        p_cam: NDArray[Shape["3"], Float],
        cam_pos: NDArray[Shape["3"], Float],
        cam_rot: NDArray[Shape["3, 3"], Float],
    ) -> NDArray[Shape["3"], Float]:
        p_cam = p_cam.reshape(3, 1)
        cam_pos = cam_pos.reshape(3, 1)
        cam_rot_inv = np.linalg.inv(cam_rot)

        p_world = cam_rot_inv @ p_cam + cam_pos

        p_world = p_world.flatten()

        return p_world


class Cam2WorldOrientationConverter:
    def __init__(self):
        pass

    def __call__(
        self,
        orientation: NDArray[Shape["3, 3"], Float],
        cam_rot: NDArray[Shape["3, 3"], Float],
    ) -> NDArray[Shape["3, 3"], Float]:
        cam_rot_inv = np.linalg.inv(cam_rot)

        orientation_world = cam_rot_inv @ orientation

        return orientation_world


class Paper2SimGraspConverter:
    def __init__(self):
        pass

    def __call__(
        self,
        grasp_paper: GraspPaperCam,
    ) -> GraspCam:
        position = (
            grasp_paper.contact_point
            + 0.5 * grasp_paper.width * grasp_paper.pose[:3, 0].flatten()
        )

        grasp_sim = GraspCam(
            score=grasp_paper.score,
            position=position,
            orientation=grasp_paper.pose[:3, :3],
            width=grasp_paper.width,
        )

        return grasp_sim


class World2ImgCoordConverter:
    def __call__(
        self,
        p_world: NDArray[Shape["3"], Float],
        cam_intrinsics: NDArray[Shape["3,3"], Float],
        cam_rot: NDArray[Shape["3,3"], Float],
        cam_pos: NDArray[Shape["3"], Float],
    ) -> NDArray[Shape["2"], Float]:
        # p_cam = R @ (p_world - T)
        # p_img_h = K @ p_cam = [[p_ix*p_cz]
        #                        [p_iy*p_cz]
        #                        [p_cz     ]]
        # p_img = [[p_ix]  = (p_img_h / p_cz)[:2] = (p_img_h / p_img_h[2])[:2]
        #           p_iy]]

        cam_pos = cam_pos.reshape((3, 1))  # (3,1)

        p_world = p_world.reshape((3, 1))  # (3,1)
        p_cam = cam_rot @ (p_world - cam_pos)
        p_img_h = cam_intrinsics @ p_cam
        p_img = (p_img_h / p_img_h[2])[:2].flatten()  # (2,)

        return p_img
