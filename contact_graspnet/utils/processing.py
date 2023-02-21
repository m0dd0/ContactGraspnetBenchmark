from pathlib import Path
from typing import Union, List

import yaml
from PIL import Image

from contact_graspnet.utils.export import Exporter
from contact_graspnet.utils.config import module_from_config
from contact_graspnet.utils.misc import get_root_dir

from contact_graspnet.utils.misc import setup_tensorflow

setup_tensorflow()

# class Compose:
#     def __init__(self, transforms: List):
#         self.transforms = transforms

#     def __call__(self, sample):
#         for transform in self.transforms:
#             sample = transform(sample)
#         return sample


def process_dataset(
    config_path: Path,
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = module_from_config(config["dataset"])
    preprocessor = dataset.transform
    model = module_from_config(config["model"])
    postprocessor = module_from_config(config["postprocessor"])
    result_path = Path(config["result_path"]).expanduser()
    exporter = Exporter(
        export_dir=result_path, move_path_contents=True, max_json_elements=1000
    )

    result_path.mkdir(parents=True, exist_ok=True)
    with open(result_path / "inference_config.yaml", "w") as f:
        yaml.dump(config, f)

    for i in range(len(dataset)):
        full_pc, segmented_pc = dataset[i]
        initial_sample = preprocessor.intermediate_results["initial_sample"]

        print(f"Processing sample {initial_sample.name}... ({i+1}/{len(dataset)})")

        network_output = (pred_grasps_cam, scores, contact_pts, widths) = model(
            full_pc, segmented_pc
        )
        grasps_cam = postprocessor(network_output)

        export_data = {
            "grasps_cam": grasps_cam,
            # "sample_segmentation": sample.segmentation,
            "sample_rgb": Image.fromarray(initial_sample.rgb),
            # "sample_depth": sample.depth,
            # "sample_intrinsics": sample.cam_intrinsics,
            "full_pc": full_pc,
            "full_pc_colors": preprocessor.intermediate_results["full_pc_colors"],
            "segmented_pc": segmented_pc,
            "segmented_pc_colors": preprocessor.intermediate_results[
                "segmented_pc_colors"
            ],
        }

        exporter(export_data, initial_sample.name)


if __name__ == "__main__":
    config_path = get_root_dir() / "configs" / "ycb.yaml"
    # config_path = get_root_dir() / "configs" / "examples.yaml"

    process_dataset(config_path)
