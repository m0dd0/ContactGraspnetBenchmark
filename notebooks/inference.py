from typing import List
from pathlib import Path

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(BASE_DIR))

from contact_graspnet.orig.contact_graspnet import config_utils
from contact_graspnet.orig.contact_graspnet.data import load_available_input_data
from contact_graspnet.orig.contact_graspnet.contact_grasp_estimator import (
    GraspEstimator,
)
from contact_graspnet.orig.contact_graspnet.visualization_utils import (
    visualize_grasps,
    show_image,
)

# import config_utils
# from data import load_available_input_data

# from contact_grasp_estimator import GraspEstimator
# from visualization_utils import visualize_grasps, show_image


def inference(
    input_paths: List[Path],
    checkpoint_dir: Path = "scene_test_2048_bs3_hor_sigma_001",
    K=None,
    local_regions: bool = False,
    skip_border_objects: bool = False,
    filter_grasps: bool = False,
    segmap_id: int = 0,
    z_range: List = (0.2, 1.8),
    forward_passes: int = 1,
):
    # Predict 6-DoF grasp distribution for given model and input data

    # :param global_config: config.yaml from checkpoint directory
    # :param checkpoint_dir: checkpoint directory
    # :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    # :param K: Camera Matrix with intrinsics to convert depth to point cloud
    # :param local_regions: Crop 3D local regions around given segments.
    # :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    # :param filter_grasps: Filter and assign grasp contacts according to segmap.
    # :param segmap_id: only return grasps from specified segmap_id.
    # :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    # :param forward_passes: Number of forward passes to run on each point cloud. Default: 1

    # load config
    global_config = config_utils.load_config(
        checkpoint_dir, batch_size=forward_passes, arg_configs=[]
    )

    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode="test")

    # Process example test scenes
    for p in input_paths:
        print(f"Loading {p}.")

        pc_segments = {}
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(
            str(p), K=K
        )

        if segmap is None and (local_regions or filter_grasps):
            raise ValueError(
                "Need segmentation map to extract local regions or filter grasps"
            )

        if pc_full is None:
            print("Converting depth to point cloud(s)...")
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
                depth,
                cam_K,
                segmap=segmap,
                rgb=rgb,
                skip_border_objects=skip_border_objects,
                z_range=z_range,
            )

        print("Generating Grasps...")
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
            sess,
            pc_full,
            pc_segments=pc_segments,
            local_regions=local_regions,
            filter_grasps=filter_grasps,
            forward_passes=forward_passes,
        )

        # Save results
        # np.savez(
        #     "results/predictions_{}".format(
        #         os.path.basename(p.replace("png", "npz").replace("npy", "npz"))
        #     ),
        #     pred_grasps_cam=pred_grasps_cam,
        #     scores=scores,
        #     contact_pts=contact_pts,
        # )

        # Visualize results
        # show_image(rgb, segmap)
        # visualize_grasps(
        #     pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors
        # )


if __name__ == "__main__":
    inference(
        input_paths=[
            Path(__file__).parent.parent
            / "contact_graspnet"
            / "data"
            / "test_data"
            / "7.npy"
        ],
        # input_paths=[flags.np_path] if not flags.png_path else [flags.png_path],
        checkpoint_dir=Path(__file__).parent.parent
        / "contact_graspnet"
        / "checkpoints"
        / "scene_test_2048_bs3_hor_sigma_001",
        K=None,
        local_regions=False,
        skip_border_objects=False,
        filter_grasps=False,
        segmap_id=0,
        z_range=[0.2, 1.8],
        forward_passes=1,
    )
