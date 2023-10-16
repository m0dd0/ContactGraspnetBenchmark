from typing import Tuple
from pathlib import Path
import os

import numpy as np
from nptyping import NDArray, Shape, Int, Float
import tensorflow.compat.v1 as tf


def posrot2pose(
    pos: NDArray[Shape["3"], Float], rot: NDArray[Shape["3,3"], Float]
) -> NDArray[Shape["4,4"], Float]:
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def pose2posrot(
    pose: NDArray[Shape["4,4"], Float]
) -> Tuple[NDArray[Shape["3"], Float], NDArray[Shape["3,3"], Float]]:
    return pose[:3, 3], pose[:3, :3]


def get_root_dir() -> Path:
    """Returns the root directory of the project aka the "contact_graspnet" directory.

    Returns:
        Path: The root directory of the project.
    """
    return Path(__file__).parent.parent


def setup_tensorflow():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    tf.disable_eager_execution()
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def exists_in_subfolder(path: Path, subfolder: Path) -> Path:
    path = Path(path)

    if not path.exists():
        path = subfolder / path

    if not path.exists():
        raise FileNotFoundError(f"Model path {path} does not exist.")

    return path
