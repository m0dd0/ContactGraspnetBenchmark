"""This file containes dataset abstractions for the different datasets used in this project.
Each dataset should be a subclass of torch.utils.data.Dataset and should implement the
__len__ and __getitem__ methods.
The __len__ method should return the number of samples in the dataset.
The __getitem__ method should return a sample from the dataset. The sample should be a
instance of the datatypes.Sample class.
The __init__ method shoulf take a transform argument which is a callable that takes a
datatypes.sample instance as input.
"""

from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from ..datatypes import YCBSimulationDataSample, OrigExampleDataSample


# class Dataset(ABC):
#     def __init__(self):
#         self._idx = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self._idx >= len(self):
#             raise StopIteration()
#         item = self[self._idx]
#         self._idx += 1
#         return item


class YCBSimulationData:
    def __init__(self, root_dir: Path, transform: Callable = None):
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(list(self.root_dir.glob("*.npz")))

    def __getitem__(self, index: int) -> YCBSimulationDataSample:
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset with length {len(self)}"
            )

        all_sample_names = [
            p.parts[-1] for p in self.root_dir.iterdir() if p.suffix == ".npz"
        ]

        all_sample_names = sorted(all_sample_names)
        sample_name = all_sample_names[index]
        sample_path = self.root_dir / sample_name

        simulation_data = np.load(sample_path)

        sample = YCBSimulationDataSample(
            rgb=simulation_data["rgb_img"],
            depth=simulation_data["depth_img"],
            points=simulation_data["point_cloud"][0],
            points_color=(simulation_data["point_cloud"][1] * 255).astype(np.uint8),
            points_segmented=simulation_data["point_cloud_seg"][0],
            points_segmented_color=(simulation_data["point_cloud_seg"][1] * 255).astype(
                np.uint8
            ),
            segmentation=simulation_data["seg_img"].astype("uint8"),
            cam_intrinsics=simulation_data["cam_intrinsics"],
            cam_pos=simulation_data["cam_pos"],
            cam_rot=Rotation.from_quat(
                simulation_data["cam_quat"][[1, 2, 3, 0]]
            ).as_matrix(),
            name=sample_name.split(".")[0],
        )

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class OrigExampleData:
    def __init__(self, root_dir: Path, transform: Callable = None):
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(list(self.root_dir.glob("*.npy")))

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset with length {len(self)}"
            )

        data = np.load(self.root_dir / f"{index}.npy", allow_pickle=True).item()

        sample = OrigExampleDataSample(
            rgb=data["rgb"],
            depth=data["depth"],
            segmentation=data["seg"],
            cam_intrinsics=data["K"],
            name=f"orig_example_{index}",
        )

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
