from pathlib import Path

class OrigExampleDataset:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def __getitem__(self, idx):