from copy import deepcopy

from contact_graspnet_benchmark.datatypes import CameraData

from . import custom_transforms as CT


class Preprocessor:
    def __init__(self, z_range):
        self.depth2point_contverter = CT.Depth2PointcloudConverter(z_range)

    def __call__(self, camera_data: CameraData):
        camera_data = deepcopy(camera_data)

        if camera_data.points is None:
            camera_data.points = self.depth2point_contverter(
                camera_data.depth, camera_data.cam_intrinsics, camera_data.rgb
            )

        return camera_data
