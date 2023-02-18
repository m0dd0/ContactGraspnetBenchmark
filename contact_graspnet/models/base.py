"""The class in this modules acts as a base class for all models in the project.
It contains useful generic functions for consructing models from config files or
from state dict paths. Other models should inherit from this class."""

from abc import ABC
from pathlib import Path

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    @staticmethod
    def check_model_path(model_path: Path) -> Path:
        model_path = Path(model_path)

        if not model_path.exists():
            model_path = Path(__file__).parent.parent / "checkpoints" / model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

        return model_path

    @classmethod
    def from_jit(cls, jit_path: Path = None, device: str = None) -> "BaseModel":
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        jit_path = jit_path or "path/to/the/default/jit/path.pth"
        jit_path = cls.check_model_path(jit_path)

        model = torch.jit.load(jit_path)
        model.to(device)

        return model

    @classmethod
    def from_state_dict_path(
        cls, state_dict_path: Path = None, device: str = None, **kwargs
    ) -> "BaseModel":
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        state_dict_path = cls.check_model_path(state_dict_path)

        model = cls(**kwargs)
        model.load_state_dict(torch.load(state_dict_path, map_location=device))

        model.to(device)

        return model

    @classmethod
    def from_config(cls, **kwargs) -> "BaseModel":
        if "jit_path" in kwargs:
            return cls.from_jit(**kwargs)

        elif "state_dict_path" in kwargs:
            return cls.from_state_dict_path(**kwargs)

        else:
            raise ValueError("No valid config for loading model.")

    # TODO define other methods here which might be shared by all models
    # e.g. save, load, error, ...
