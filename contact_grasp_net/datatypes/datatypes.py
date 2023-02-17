"""This module contains only dataclasses. These dataclasses hold only information and no
logic. They are used to define the datatypes used in the project.
The datatypes can be grouped into two categories:
- DatasetSamples: These are the samples that are returned by the datasets
There should be an compatible preprocessing pipeline which accepts these samples as input
in order to used them in the project.
- Results: These are the results that are returned by the posprocessing steps
There should be an compatible postprocessing pipeline which outputs these dataclasses as results.
"""
from dataclasses import dataclass
from abc import ABC


@dataclass
class DatasetSample(ABC):
    name: str


@dataclass
class YCBSimulationDataSample(DatasetSample):
    attr1: ...
    attr2: ...
    attr3: ...

# ... dataclasses for other datasets here

@dataclass
class ResultBase(ABC):
    pass

@dataclass
class SomeResult(ResultBase):
    attr1: ...
    attr2: ...
    attr3: ...

# ... dataclasses for other results here