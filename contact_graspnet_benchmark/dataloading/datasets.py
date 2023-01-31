from pathlib import Path

import numpy as np

from contact_graspnet_benchmark.datatypes import CameraData


class OrigExampleDataset:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def __getitem__(self, idx: int):
        segmap, rgb, depth, pc_full, pc_colors = None, None, None, None, None

        # if K is not None:
        #     if isinstance(K,str):
        #         cam_K = eval(K)
        #     cam_K = np.array(K).reshape(3,3)

        # if '.np' in p:
        sample_path = self.dataset_dir / f"{idx}.npy"
        assert sample_path.exists()

        data = np.load(sample_path, allow_pickle=True)
        data = data.item()

        return CameraData(
            rgb=data["rgb"],
            depth=data["depth"],
            points=None,  # data["xyz"],
            points_colors=None,  # data["xyz_color"]
            segmentation=data["seg"],
            name=f"exaple_{idx}",
        )