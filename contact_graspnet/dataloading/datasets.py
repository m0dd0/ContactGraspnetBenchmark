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

from torch.utils.data import Dataset

from ..datatypes import YCBSimulationDataSample

class YCBSimulationData(Dataset):
    def __init__(self, root: Path, transform: Callable=None):
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        # TODO: Implement this method
        raise NotImplementedError()


    def __getitem__(self, index: int) -> YCBSimulationDataSample:
        # TODO: Implement this method
        raise NotImplementedError()

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
