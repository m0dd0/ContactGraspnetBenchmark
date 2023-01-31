from nptyping import NDArray, Shape, Float

from contact_graspnet_benchmark.orig.contact_graspnet.data import depth2pc


class Depth2PointcloudConverter:
    def __init__(self, z_range):
        self.z_range = z_range

    def __call__(self, depth: NDArray[Shape["1,H,W"], Float], K, rgb=None):
        pc_full, pc_colors = depth2pc(depth, K, rgb)

        # Threshold distance
        if pc_colors is not None:
            pc_colors = pc_colors[
                (pc_full[:, 2] < self.z_range[1]) & (pc_full[:, 2] > self.z_range[0])
            ]
        pc_full = pc_full[
            (pc_full[:, 2] < self.z_range[1]) & (pc_full[:, 2] > self.z_range[0])
        ]

        return pc_full, pc_colors


# class PointcloudFilter:
#     def __init__(self, z_range, skip_border_objects: bool = False, margin_px: int = 5):
#         self.z_range = z_range
#         self.skip_border_objects = skip_border_objects
#         self.margin_px = margin_px

#     def __call__(self, pc, segmap):
#         pc_segments = {}

#         obj_instances = np.unique(segmap[segmap > 0])

#         for i in obj_instances:

#             if self.skip_border_objects:
#                 obj_i_y, obj_i_x = np.where(segmap == i)
#                 if (
#                     np.any(obj_i_x < self.margin_px)
#                     or np.any(obj_i_x > segmap.shape[1] - self.margin_px)
#                     or np.any(obj_i_y < self.margin_px)
#                     or np.any(obj_i_y > segmap.shape[0] - self.margin_px)
#                 ):
#                     # print("object {} not entirely in image bounds, skipping".format(i))
#                     continue

#             inst_mask = segmap == i
#             pc_segment, _ = depth2pc(depth * inst_mask, K)
#             pc_segments[i] = pc_segment[
#                 (pc_segment[:, 2] < z_range[1]) & (pc_segment[:, 2] > z_range[0])
#             ]  # regularize_pc_point_count(pc_segment, grasp_estimator._contact_grasp_cfg['DATA']['num_point'])
