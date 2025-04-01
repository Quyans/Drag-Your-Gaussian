#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from modules.scene.dataset_readers import sceneLoadTypeCallbacks
from modules.renderers.gaussians_renderer import Simple_Camera

def cameraList_load(cam_infos, h, w, render_res_factor):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(
            Simple_Camera(colmap_id=c.uid, R=c.R, T=c.T,
                   FoVx=c.FovX, FoVy=c.FovY, h=h, w=w, qvec = c.qvec,
                   image_name=c.image_name, uid=id, data_device='cuda',
                   render_res_factor=render_res_factor)
        )
    return camera_list

class CamScene:
    def __init__(self, source_path, h=512, w=512, aspect=-1,render_res_factor=1):
        """b
        :param path: Path to colmap scene main folder.
        """
        if aspect != -1:
            h = 512
            w = 512 * aspect

        if os.path.exists(os.path.join(source_path, "sparse")):
            if h == -1 or w == -1:
                scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, None, False)
                if scene_info.train_cameras[0].height > scene_info.train_cameras[0].width:
                    #upsample factor
                    h = 512 * render_res_factor
                    w = int(scene_info.train_cameras[0].width / scene_info.train_cameras[0].height * 512 / 8) * 8 * render_res_factor
                else:
                    w = 512 * render_res_factor
                    h = int(scene_info.train_cameras[0].height / scene_info.train_cameras[0].width * 512 / 8) * 8 * render_res_factor
                # h = scene_info.train_cameras[0].height
                # w = scene_info.train_cameras[0].width
                # if w > 1920:
                #     scale = w / 1920
                #     h /= scale
                #     w /= scale
            else:
                h = h * render_res_factor
                w = w * render_res_factor
                scene_info = sceneLoadTypeCallbacks["Colmap_hw"](source_path, h, w, None, False)

        else:
            assert False, "Could not recognize scene type!"

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        self.cameras = cameraList_load(scene_info.train_cameras, h, w, render_res_factor)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

