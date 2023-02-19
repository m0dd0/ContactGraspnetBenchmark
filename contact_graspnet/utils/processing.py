from pathlib import Path
from typing import Union

import yaml
from PIL import Image

from contact_graspnet.dataloading import OrigExampleData, YCBSimulationData
from contact_graspnet.utils.visualization import mlab_pose_vis
from contact_graspnet.utils.export import Exporter
from contact_graspnet.utils.config import module_from_config
from contact_graspnet.utils.misc import get_root_dir

from contact_graspnet.utils.misc import setup_tensorflow

setup_tensorflow()


def process_dataset(
    dataset: Union[OrigExampleData, YCBSimulationData],
    result_path: Path,
    config_path: Path,
    visualize: bool = False,
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    preprocessor = module_from_config(config["preprocessor"])
    model = module_from_config(config["model"])
    postprocessor = module_from_config(config["postprocessor"])
    exporter = Exporter(export_dir=result_path, move_path_contents=True)

    config["processed_datset"] = str(dataset.root_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    with open(result_path / "inference_config.yaml", "w") as f:
        yaml.dump(config, f)

    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Processing sample {sample.name}... ({i+1}/{len(dataset)})")

        pointcloud = preprocessor(sample)
        network_output = (pred_grasps_cam, scores, contact_pts) = model(pointcloud)
        grasps_img = postprocessor(network_output)

        vis_temp_path = Path.cwd() / "temp.png"
        if visualize:
            mlab_pose_vis(
                pointcloud,
                grasps_img,
                preprocessor.intermediate_results["pointcloud_colors"],
                image_path=vis_temp_path,
            )

        export_data = {
            "mlab_vis.png": vis_temp_path if visualize else None,
            "grasps_img": grasps_img,
            # we also export the sample data
            "sample_segmentation": sample.segmentation,
            "sample_rgb": Image.fromarray(sample.rgb),
            "sample_depth": sample.depth,
            "sample_intrinsics": sample.cam_intrinsics,
        }

        exporter(export_data, sample.name)


if __name__ == "__main__":
    process_dataset(
        dataset=OrigExampleData(get_root_dir() / "data" / "raw" / "orig_test_data"),
        result_path=get_root_dir() / "data" / "results" / "orig_test_data",
        config_path=get_root_dir() / "configs" / "default_example_inference.yaml",
        visualize=True,
    )

    # process_dataset(
    #     dataset=YCBSimulationData(Path.home() / "Documents" / "ycb_sim_data_1"),
    #     result_path=get_root_dir() / "data" / "results" / "ycb_sim_data_1",
    #     config_path=get_root_dir() / "configs" / "default_ycb_inference.yaml",
    #     visualize=True,
    # )

