"""The class in this modules acts as a base class for all models in the project.
It contains useful generic functions for consructing models from config files or
from state dict paths. Other models should inherit from this class."""

from abc import ABC


class BaseModel(ABC):
    # the teplate code is torch specific but we use tensorflow in this project
    pass
