from pathlib import Path
from typing import Dict, Any

import tensorflow.compat.v1 as tf

from contact_graspnet_benchmark.orig.contact_graspnet import config_utils
from contact_graspnet_benchmark.orig.contact_graspnet.contact_grasp_estimator import (
    GraspEstimator,
)
from contact_graspnet_benchmark.datatypes import CameraData


class End2EndProcessor:
    def __init__(
        self,
        checkpoint_dir: Path = None,
        forward_passes: int = 1,
    ):
        self.forward_passes = forward_passes

        # setup grasp estimator
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        self.grasp_estimator = GraspEstimator(
            config_utils.load_config(checkpoint_dir, batch_size=self.forward_passes),
            sess,
        )
        self.grasp_estimator.build_network()
        self.grasp_estimator.load_weights(
            tf.train.Saver(save_relative_paths=True),
            checkpoint_dir,
            mode="test",
        )

    def __call__(self, points):
        (
            pred_grasps_cam,
            scores,
            contact_pts,
            _,
        ) = self.grasp_estimator.predict_scene_grasps(
            points,
            # pc_segments=pc_segments,
            # local_regions=local_regions,
            # filter_grasps=filter_grasps,
            forward_passes=self.forward_passes,
        )

        return pred_grasps_cam, scores, contact_pts
