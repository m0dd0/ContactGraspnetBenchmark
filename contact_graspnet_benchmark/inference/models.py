import tensorflow.compat.v1 as tf

from contact_graspnet_benchmark.orig.contact_graspnet import config_utils
from contact_graspnet_benchmark.orig.contact_graspnet.contact_grasp_estimator import (
    GraspEstimator,
)


class WrappedGraspEstimator:
    def __init__(self, checkpoint_dir, local_regions, filter_grasps):
        # TODO handle this in postprocessor
        self.local_regions = local_regions
        self.filter_grasps = filter_grasps

        tf.disable_eager_execution()
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # setup grasp estimator
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        self.grasp_estimator = GraspEstimator(
            config_utils.load_config(checkpoint_dir),
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
            predicted_grasps_cam,
            scores,
            contact_points,
            _,
        ) = self.grasp_estimator.predict_scene_grasps(
            points,
            # pc_segments=pc_segments, # TODO
            local_regions=self.local_regions,
            filter_grasps=self.filter_grasps,
            forward_passes=self.forward_passes,
        )

        return predicted_grasps_cam, scores, contact_points
