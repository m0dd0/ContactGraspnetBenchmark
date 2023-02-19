"""This module contains the custom transforms that are used in the postprocessing pipelines.
They should be as concise as possible and only contain the logic that is necessary to
execute a singe transformation step.
They might also be used directly in a Compose to make a descriptive pipeline.
"""
from typing import List


from contact_graspnet.datatypes import GraspImg


class TopScoreFilter:
    def __init__(self, top_k: int = 1):
        self.top_k = top_k

    def __call__(self, grasps: List[GraspImg]) -> List[GraspImg]:
        if len(grasps) <= self.top_k:
            return grasps

        return sorted(grasps, key=lambda grasp: grasp.score, reverse=True)[: self.top_k]
