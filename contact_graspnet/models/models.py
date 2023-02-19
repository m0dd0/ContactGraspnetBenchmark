""" This module contains the network architecture(s) of the project."""

from typing import Tuple
from pathlib import Path

import tensorflow.compat.v1 as tf
from nptyping import NDArray, Shape, Float, Int
import yaml

# from .custom_modules import CustomModuleExample
from .base import BaseModel

from contact_graspnet.orig.contact_graspnet.contact_grasp_estimator import (
    GraspEstimator,
)


class ContactGraspnet(BaseModel):
    def __init__(self, config_path: Path, checkpoint_dir: Path, batch_size=1):
        super().__init__()

        config_path = Path(config_path).expanduser()
        checkpoint_dir = Path(checkpoint_dir).expanduser()

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
        self, pc_full: NDArray[Shape["N, 3"], Float]
    ) -> Tuple[
        NDArray[Shape["N,4,4"], Float],
        NDArray[Shape["N"], Float],
        NDArray[Shape["N, 3"], Float],
        NDArray[Shape["N"], Float],
    ]:
        # we never do segmentation/local regions in this model, this should be done in the preprocessing
        (
            pred_grasps_cam,
            scores,
            contact_pts,
            gripper_openings,
        ) = self._grasp_estimator.predict_scene_grasps(
            self._sess,
            pc_full
            # pc_segments=pc_segments,
            # local_regions=False,
            # filter_grasps=False,
            # forward_passes=1,
        )

        assert len(pred_grasps_cam) == 1
        assert len(scores) == 1
        assert len(contact_pts) == 1

        pred_grasps_cam = pred_grasps_cam[-1]
        scores = scores[-1]
        contact_pts = contact_pts[-1]

        return pred_grasps_cam, scores, contact_pts, gripper_openings
