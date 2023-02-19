""" This modules contains the postprocessing pipelines. A postprocessing pipeline should
accept the output of the model (i.e. a tensor) and return a usable, interpretable result.
The result type should be a subclass of ResultBase.
The postprocessing pipeline should also store intermediate results in the
intermediate_results dictionary. These results can be used in the end for closer evauation
and debugging.
Since a pipeline is not used in backpropagation it is not necessary to implement it as
a torch.nn.Module.
A pipeline should consist out of multiple submodules that are called in a specific order.
For clarity it should be avoided to have logic inthe pipeline itself. Instead the logic
should be implemented in the submodules. The pipeline itself should only manage the
flow of information through the submodules. If a pipeline has no or only a single
submodule it might be more suitable to implement it as a submodule instead of a pipeline
for improved reusability. It is also good practice to have the pipeline accept (multiple)
submodules as arguments. This way the pipeline can be used with different submodules
which increases modularity.
Somteimes it might be useful to have a subpipeline that is used in multiple pipelines.
This subpipeline will output an intermediate result that is used in multiple pipelines
but not the final result. Keep in mind that you need to manage the intermediate result 
of the subpipeline yourself.
Also it might be usefule to have further pipelines which process the result further.
These pipelines might take an ResultBase as input and process it further with more
information etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List

from nptyping import NDArray, Shape, Float

from contact_graspnet.datatypes import ResultBase, GraspCam
from . import custom_transforms as CT


class PostprocessorBase(ABC):
    def __init__(self):
        super().__init__()
        self.intermediate_results: Dict[str, Any] = {}

    @abstractmethod
    def __call__(
        self,
        network_output: Tuple[
            NDArray[Shape["N,4,4"], Float],
            NDArray[Shape["N"], Float],
            NDArray[Shape["N, 3"], Float],
        ],
    ) -> List[ResultBase]:
        pass


class Postprocessor(PostprocessorBase):
    def __init__(self, top_score_filter: CT.TopScoreFilter):
        super().__init__()

        self.top_score_filter = top_score_filter

    def __call__(
        self,
        network_output: Tuple[
            NDArray[Shape["N,4,4"], Float],
            NDArray[Shape["N"], Float],
            NDArray[Shape["N, 3"], Float],
        ],
    ) -> List[GraspCam]:
        grasps_cam = []
        for pose, score, contact_point in zip(
            network_output[0], network_output[1], network_output[2]
        ):
            grasps_cam.append(
                GraspCam(
                    score=score,
                    contact_point=contact_point,
                    pos=pose[:3, 3],
                    orientation=pose[:3, :3],
                )
            )

        self.intermediate_results["all_grasps"] = grasps_cam

        top_grasps = grasps_cam
        if self.top_score_filter is not None:
            top_grasps = self.top_score_filter(grasps_cam)

        return top_grasps
