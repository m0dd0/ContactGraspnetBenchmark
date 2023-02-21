from pathlib import Path
from typing import Union, List

import yaml
from PIL import Image

from contact_graspnet.preprocessing import BinarySegmentationSample
from contact_graspnet.dataloading import OrigExampleData, YCBSimulationData
from contact_graspnet.utils.export import Exporter
from contact_graspnet.utils.config import module_from_config
from contact_graspnet.utils.misc import get_root_dir

from contact_graspnet.utils.misc import setup_tensorflow

setup_tensorflow()


def process_dataset(
    dataset: Union[OrigExampleData, YCBSimulationData],
    result_path: Path,
    config_path: Path,
    skip_obj_ids: List[int] = None,
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    preprocessor = module_from_config(config["preprocessor"])
    model = module_from_config(config["model"])
    postprocessor = module_from_config(config["postprocessor"])
    exporter = Exporter(
        export_dir=result_path, move_path_contents=True, max_json_elements=1000
    )

    config["processed_datset"] = str(dataset.root_dir)
    config["skip_obj_ids"] = skip_obj_ids
    result_path.mkdir(parents=True, exist_ok=True)
    with open(result_path / "inference_config.yaml", "w") as f:
        yaml.dump(config, f)

    for i in range(len(dataset)):
        if skip_obj_ids is not None and i in skip_obj_ids:
            print(f"Skipping sample {i}... ({i+1}/{len(dataset)})")
            continue

        sample = dataset[i]
        print(f"Processing sample {sample.name}... ({i+1}/{len(dataset)})")

        full_pc, segmented_pc = preprocessor(sample)
        network_output = (pred_grasps_cam, scores, contact_pts, widths) = model(
            full_pc, segmented_pc
        )
        grasps_cam = postprocessor(network_output)

        export_data = {
            "grasps_cam": grasps_cam,
            # "sample_segmentation": sample.segmentation,
            "sample_rgb": Image.fromarray(sample.rgb),
            # "sample_depth": sample.depth,
            # "sample_intrinsics": sample.cam_intrinsics,
            "full_pc": full_pc,
            "full_pc_colors": preprocessor.intermediate_results["full_pc_colors"],
            "segmented_pc": segmented_pc,
            "segmented_pc_colors": preprocessor.intermediate_results[
                "segmented_pc_colors"
            ],
        }

        exporter(export_data, sample.name)


if __name__ == "__main__":
    # segmentation_id = 4.0
    # process_dataset(
    #     dataset=OrigExampleData(
    #         get_root_dir() / "data" / "raw" / "orig_test_data",
    #         transform=BinarySegmentationSample(segmentation_id),
    #     ),
    #     result_path=get_root_dir()
    #     / "data"
    #     / "results"
    #     / f"orig_test_data_seg{int(segmentation_id)}",
    #     config_path=get_root_dir() / "configs" / "default_inference.yaml",
    # )

    skip_obj_ids = [17, 18, 19, 22, 23, 34, 36, 50, 51, 52, 74]
    i = 3
    process_dataset(
        dataset=YCBSimulationData(Path.home() / "Documents" / f"ycb_sim_data_{i}"),
        result_path=get_root_dir() / "data" / "results" / f"ycb_sim_data_{i}",
        config_path=get_root_dir() / "configs" / "default_inference.yaml",
        skip_obj_ids=skip_obj_ids,
    )
