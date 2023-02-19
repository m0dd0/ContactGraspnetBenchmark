from pathlib import Path

import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
import numpy as np
import yaml

from contact_graspnet.dataloading import OrigExampleData

from contact_graspnet.utils.visualization import mlab_pose_vis
from contact_graspnet.utils.export import Exporter
from contact_graspnet.utils.misc import get_root_dir
from contact_graspnet.utils.config import module_from_config 

def process_example_data(datset_path: Path, result_path: Path, config_path: Path):
    pass