""" This module contains the network architecture(s) of the project."""

from typing import Tuple
from pathlib import Path

import tensorflow.compat.v1 as tf
from nptyping import NDArray, Shape, Float, Int
import yaml
import numpy as np

# from .custom_modules import CustomModuleExample
from .base import BaseModel

from contact_graspnet.orig.contact_graspnet.contact_grasp_estimator import (
    GraspEstimator,
)
from contact_graspnet.utils.misc import exists_in_subfolder, get_root_dir


class ContactGraspnet(BaseModel):
    def __init__(self, config_path: Path, checkpoint_dir: Path, batch_size=1):
        super().__init__()

        config_path = Path(config_path).expanduser()
        checkpoint_dir = Path(checkpoint_dir).expanduser()

        checkpoint_dir = exists_in_subfolder(
            checkpoint_dir, get_root_dir() / "checkpoints"
        )
        config_path = exists_in_subfolder(config_path, checkpoint_dir)

        # TODO recator grasp estimator

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config["OPTIMIZER"]["batch_size"] = batch_size

        self._grasp_estimator = GraspEstimator(config)
        self._grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self._sess = tf.Session(config=config)

        self._grasp_estimator.load_weights(
            self._sess, saver, checkpoint_dir, mode="test"
        )

    def __call__(
        self,
        pc_full: NDArray[Shape["N, 3"], Float],
        pc_segment: NDArray[Shape["N, 3"], Float] = None,
    ) -> Tuple[
        NDArray[Shape["N,4,4"], Float],
        NDArray[Shape["N"], Float],
        NDArray[Shape["N, 3"], Float],
        NDArray[Shape["N"], Float],
    ]:
        # the processing of partial segmented pointlcouds is a mess in the original code
        # therefore we did not put it into the preprocessing pipeline and rather adapted
        # the model wrapper to handle it

        pc_segments = {-1: pc_segment} if pc_segment is not None else {}
        local_regions = pc_segment is not None
        filter_grasps = pc_segment is not None

        (
            pred_grasps_cam,
            scores,
            contact_pts,
            gripper_openings,
        ) = self._grasp_estimator.predict_scene_grasps(
            self._sess,
            pc_full,
            pc_segments=pc_segments,
            local_regions=local_regions,
            filter_grasps=local_regions,
            # forward_passes=1,
        )

        assert (
            len(pred_grasps_cam)
            == len(scores)
            == len(contact_pts)
            == len(gripper_openings)
            in (0, 1)
        )

        if len(pred_grasps_cam) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        pred_grasps_cam = pred_grasps_cam[-1]
        scores = scores[-1]
        contact_pts = contact_pts[-1]
        gripper_openings = gripper_openings[-1]

        # this is bug in the original code: if only one grasps gets predicted, the width output is not a array but a single float
        if gripper_openings.ndim == 0:
            gripper_openings = np.array([gripper_openings])

        return pred_grasps_cam, scores, contact_pts, gripper_openings
