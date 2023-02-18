"""This module contains utilities to visualize the different data created during processing.
In contrast to the other parts of this project the visualization utitlities are not
designed to be used as a pipeline/processor. Instead functions are provided that can be
used to visualize the data in a notebook or in a script.
Try to use functions which operate on matplotlib.Axes objects as this makes it easier
to combine the different visualizations.
"""

from typing import List

from matplotlib import pyplot as plt
import numpy as np

from contact_graspnet.datatypes import ImageGrasp, RealGrasp
from contact_graspnet.utils.geometry import get_antipodal_points


def make_tensor_displayable(
    tensor, convert_chw: bool = False, convert_to_int: bool = False
):
    """Executes different operations to make a tensor displayable.
    The tensor is always converted to a numpy array and squeezed.

    Args:
        tensor: A datastrucutre that can be converted to a numpy array.
        convert_chw (bool, optional): Converter a chw tensor to a hwc tensor.
            Defaults to False.
        convert_to_int (bool, optional): Converts the datatype to uint8. Defaults to False.

    Returns:
        _type_: _description_
    """
    tensor = np.array(tensor)
    tensor = np.squeeze(tensor)  # convert one channel images to 2d tensors
    assert len(tensor.shape) in [2, 3], "squeezed Tensor must be 2 or 3 dimensional"

    if convert_chw:
        assert tensor.shape[0] == 3, "first dimension must be 3"
        # chw -> hwc
        tensor = np.transpose(tensor, (1, 2, 0))

    if convert_to_int:
        tensor = tensor.astype("uint8")

    return tensor


def overview_fig(
    # different data to visualize
    fig=None,
):
    if fig is None:
        fig = plt.figure()

    # TODO add different data to visualize

    raise NotImplementedError()

    return fig
