""" This module contains the network architecture(s) of the project."""

import tensorflow.compat.v1 as tf
from nptyping import NDArray, Shape, Float, Int
import yaml

# from .custom_modules import CustomModuleExample
from .base import BaseModel

from contact_graspnet.orig.contact_graspnet.contact_grasp_estimator import (
    GraspEstimator,
)


class ContactGraspnet(BaseModel):
    def __init__(self, config_path, checkpoint_dir, batch_size=1):
        super().__init__()

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

    def __call__(self, pc_full: NDArray[Shape["N, 3"], Float]) -> "Tensor[...]":
        (
            pred_grasps_cam,
            scores,
            contact_pts,
            _,
        ) = self._grasp_estimator.predict_scene_grasps(
            self._sess,
            pc_full
            # pc_segments=pc_segments,
            # local_regions=False,
            # filter_grasps=False,
            # forward_passes=1,
        )

        return pred_grasps_cam, scores, contact_pts
