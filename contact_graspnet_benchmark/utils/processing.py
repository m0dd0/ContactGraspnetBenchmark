from pathlib import Path
from typing import Dict, Any


from contact_graspnet_benchmark.preprocessing import Preprocessor
from contact_graspnet_benchmark.inference import GraspEstimator
from contact_graspnet_benchmark.datatypes import CameraData


class End2EndProcessor:
    def __init__(
        self,
        preprocessor: Preprocessor=None,
        model: GraspEstimator=None,
        postprocessor=None,
        visualizer=None,
        exporter=None,
    ):
        if preprocessor is None:
            preprocessor = Preprocessor()
        self.preprocessor = preprocessor

        if model is None:
            model = GraspEstimator()

    def __call__(self, sample: CameraData):
        model_input = self.preprocessor(sample)

        
        
