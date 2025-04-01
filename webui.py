import time
import copy
import numpy as np
import torch
import torchvision
import rembg
import ast
from modules.scene.colmap_loader import qvec2rotmat
from modules.renderers.gaussians_renderer import Simple_Camera
from modules.optimizers.gs_utils import load_ply
from modules.optimizers.gs_optimizer import GSOptimizer
from modules.renderers.gaussians_renderer import GaussianRenderer
from utils import sample_from_dense_cameras, inverse_sigmoid, export_ply_for_gaussians, export_points_for_gaussians

from torchvision.ops import masks_to_boxes
from modules.utils.graphics_utils import fov2focal

import viser
import viser.transforms as tf
from dataclasses import dataclass, field
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from kornia.geometry.quaternion import Quaternion

from omegaconf import OmegaConf

from argparse import ArgumentParser

from modules.scene.camera_scene import CamScene
import math

import os
import random
import datetime
import subprocess
from pathlib import Path
import json


class WebUI:
    def __init__(self, args, opt) -> None:
        self.args = args
        self.opt = opt
        self.device  = f'cuda:{args.gpu}'
        self.gs_source = args.gs_source
        self.colmap_dir = args.colmap_dir
        self.out_dir = args.out_dir
        self.port = 8888
        # training cfg

        self.tracing_point_start_3d = None
        self.tracing_point_end_3d = None
        
        self.handle_points = []
        self.target_points = []

        # front end related
        self.colmap_cameras = None
        self.render_cameras = None

        self.training = False

        # load
        self.gaussian = load_ply(self.gs_source)

        self.edit_mask = torch.ones((self.gaussian[0].shape[0])).bool().to(self.device)

        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=-1, w=-1, render_res_factor=self.opt['scene']['render_res_factor'])
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras


        self.gsoptimizer = GSOptimizer(**self.opt['gsoptimizer']['args'],
                                       image_height=self.colmap_cameras[0].image_height//self.colmap_cameras[0].render_res_factor,
                                       image_width=self.colmap_cameras[0].image_width//self.colmap_cameras[0].render_res_factor,
                                       train_args=args,
                                       ).to(self.device)
        

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.origin_frames = {}

        #self.parser = ArgumentParser(description="Training script parameters")
        #self.pipe = PipelineParams(self.parser)

        # status

        self.viewer_need_update = False
        
        self.filter_gs_switch = False

        self.server = viser.ViserServer(port=self.port)
        self.draw_flag = True
        self.point_tracing_draw_flag = True
        with self.server.add_gui_folder("Render Setting"):
            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=4096, step=2, initial_value=2048
            )

            self.FoV_slider = self.server.add_gui_slider(
                "FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1
            )

            self.fps = self.server.add_gui_text(
                "FPS", initial_value="-1", disabled=True
            )
            self.save_button = self.server.add_gui_button("Save Gaussian")
            self.save_mask_and_point = self.server.add_gui_button("Save Mask&Point")

            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False
            )

        with self.server.add_gui_folder("Filter Setting"):
            self.filter_enabled = self.server.add_gui_checkbox(
                "Enable Filter",
                initial_value=False,
            )
            self.draw_filter_bbox = self.server.add_gui_checkbox(
                "Draw Filter Box", initial_value=False
            )
            self.alpha_blend_slider = self.server.add_gui_slider(
                "alpha", min=0.0, max=0.05,step=0.001, initial_value=0.01
            )
            self.filter_gs_bt = self.server.add_gui_button("Filter Now!")
            self.filter_gs_reset = self.server.add_gui_button("Reset Filter!")

        
        with self.server.add_gui_folder("Drag Setting"):
            self.drag_point_enabled = self.server.add_gui_checkbox(
                "Drag Point Enalbled",
                initial_value=False,
            )
            self.add_tracing_point = self.server.add_gui_checkbox(
                "Add Point",
                initial_value=False,
            )
            self.add_point_to_text_button = self.server.add_gui_button("Add Drag Point To Text")

            self.handle_points_text = self.server.add_gui_text(
                "handle_point",
                initial_value="",
                visible=True,
            )
            self.target_points_text = self.server.add_gui_text(
                "target_point",
                initial_value="",
                visible=True,
            )
            self.input_point_from_text_button = self.server.add_gui_button("Input Drag Point")
            self.clear_point_button = self.server.add_gui_button("Clear Drag Point")

            self.drag_now_button = self.server.add_gui_button("Drag Now!!")

        self.left_up = self.server.add_gui_vector2(
            "Left UP",
            initial_value=(0, 0),
            step=1,
            visible=False,
        )
        self.right_down = self.server.add_gui_vector2(
            "Right Down",
            initial_value=(0, 0),
            step=1,
            visible=False,
        )
        # 追踪的2d起始点
        self.tracing_point_start_2d = self.server.add_gui_vector2(
            "Point_Start",
            initial_value=(0, 0),
            step=1,
            visible=False,
        )
        self.tracing_point_end_2d = self.server.add_gui_vector2(
            "Point End",
            initial_value=(0, 0),
            step=1,
            visible=False,
        )
        
        @self.save_button.on_click
        def _(_):
            print("Saving Gaussian to "+f"{self.out_dir}/gaussian.ply")
            export_ply_for_gaussians(f"{self.out_dir}/gaussian", self.gaussian)

        @self.save_mask_and_point.on_click
        def _(_):
            export_dir = os.path.join(self.out_dir,"export")
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            print("Saving Gaussian Mask to " + os.path.join(export_dir,"gaussian_mask.pt"))
            torch.save(self.edit_mask.detach().cpu(), os.path.join(export_dir,"gaussian_mask.pt"))

            print("Saving Drag Point to " + os.path.join(export_dir,"drag_points.json"))
            with open(os.path.join(export_dir,"drag_points.json"), "w", encoding="utf-8") as f:
                json.dump({"handle_points": self.handle_points, "target_points": self.target_points}, 
                          f, ensure_ascii=False, indent=4)

        @self.filter_gs_bt.on_click
        def _(_):
            self.filter_gs_switch = True
        
        @self.filter_gs_reset.on_click
        def _(_):
            self.filter_gs_switch = False
            self.edit_mask = torch.ones((self.gaussian[0].shape[0])).bool().to(self.device)
        

        with torch.no_grad():
            self.frames = []
            random.seed(0)
            frame_index = random.sample(
                range(0, len(self.colmap_cameras)),
                min(len(self.colmap_cameras), 20),
            )
            for i in frame_index:
                self.make_one_camera_pose_frame(i)

        @self.frame_show.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.frame_show.value
            self.server.world_axes.visible = self.frame_show.value

        @self.server.on_scene_click
        def _(pointer):
            self.click_cb(pointer)

        
        @self.add_point_to_text_button.on_click
        def _(_):
            #self.handle_points.append(self.tracing_point_start_3d)
            #self.target_points.append(self.tracing_point_end_3d)
            handle_point_list = [] if self.handle_points_text.value == '' else ast.literal_eval(self.handle_points_text.value)
            handle_point_list.append(self.tracing_point_start_3d[0].tolist())
            self.handle_points_text.value = str(handle_point_list)

            target_point_list = [] if self.target_points_text.value == '' else ast.literal_eval(self.target_points_text.value)
            target_point_list.append(self.tracing_point_end_3d[0].tolist())
            self.target_points_text.value = str(target_point_list)

        @self.input_point_from_text_button.on_click
        def _(_):
            self.handle_points = ast.literal_eval(self.handle_points_text.value)
            self.target_points = ast.literal_eval(self.target_points_text.value)

        @self.clear_point_button.on_click
        def _(_):
            self.handle_points = []
            self.target_points = []

        @self.drag_now_button.on_click
        def _(_):
            edit_mask = self.edit_mask

            xyz, features, opacity, scales, rotations = self.gaussian
            gaussians_mask = (xyz[edit_mask],features[edit_mask],opacity[edit_mask],scales[edit_mask],rotations[edit_mask])

            #delete invaild camera pose
            self.train_colmap_cameras = []
            for i,cam in enumerate(self.colmap_cameras):
                #rendered_image, rendered_depth, rendered_mask = refiner.renderer.render(*gaussians, cam, scaling_modifier=1.0, bg_color=None)
                rendered_image, rendered_depth, rendered_mask = self.gsoptimizer.renderer([cam],self.gaussian, scaling_modifier=1.0, bg_color=None)

                assert len(self.handle_points) == len(self.target_points)
                vaild_flag = True
                for j in range(len(self.handle_points)):
                    h_pixel,h_depth = cam.world_to_pixel(self.handle_points[j])
                    g_pixel,g_depth = cam.world_to_pixel(self.target_points[j])
                    if h_depth is None or g_depth is None: 
                        # or h_depth > rendered_depth[0,0,int(h_pixel[1]),int(h_pixel[0])] + 1 or \
                        # g_depth > rendered_depth[0,0,int(g_pixel[1]),int(g_pixel[0])] + 1:
                        vaild_flag = False

                rendered_image, rendered_depth, rendered_mask = self.gsoptimizer.renderer([cam],self.gaussian, scaling_modifier=1.0, bg_color=None)
                rendered_image_mask, rendered_depth_mask, _ = self.gsoptimizer.renderer([cam],gaussians_mask, scaling_modifier=1.0, bg_color=None)
                mask_image = torch.ones_like(rendered_depth_mask)
                mask_image[rendered_depth_mask==cam.zfar] = 0
                mask_image[rendered_depth<rendered_depth_mask - self.opt['gsoptimizer']['args']['mask_depth_treshold']] = 0
                if (mask_image==1).sum() < 50:
                    vaild_flag = False

                if vaild_flag:
                    self.train_colmap_cameras.append(cam)
            
            #valid camera number must > 10
            assert len(self.train_colmap_cameras) > 10

            #output init image and point
            for i,cam in enumerate(self.train_colmap_cameras):
                #rendered_image, rendered_depth, rendered_mask = refiner.renderer.render(*gaussians, cam, scaling_modifier=1.0, bg_color=None)
                rendered_image, rendered_depth, rendered_mask = self.gsoptimizer.renderer([cam], self.gaussian, scaling_modifier=1.0, bg_color=None)

                import cv2
                img = (rendered_image/2 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                img = (img.clip(min = 0, max = 1)*255.0).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                assert len(self.handle_points) == len(self.target_points)
                for j in range(len(self.handle_points)):
                    h_pixel,_ = cam.world_to_pixel(self.handle_points[j])
                    g_pixel,_ = cam.world_to_pixel(self.target_points[j])
                    cv2.circle(img, (int(h_pixel[0]),int(h_pixel[1])), 8, (255, 0, 0), -1)
                    cv2.circle(img, (int(g_pixel[0]),int(g_pixel[1])), 8, (0, 0, 255), -1)
                    cv2.arrowedLine(img, (int(h_pixel[0]),int(h_pixel[1])), 
                                         (int(g_pixel[0]),int(g_pixel[1])), (255, 255, 255), 2, tipLength=0.5)
                    # color = (0,0,255) if j == 0 else ((0,255,0) if j == 1 else (255,0,0))
                    # cv2.arrowedLine(img, (int(h_pixel[0]),int(h_pixel[1])), 
                    #                     (int(g_pixel[0]),int(g_pixel[1])), color)
                init_dir = os.path.join(self.out_dir,"init")
                if not os.path.exists(init_dir):
                    os.makedirs(init_dir)
                cv2.imwrite(f"{init_dir}/cam{i+1}_{cam.image_name}.png",img)
                rendered_image_mask, rendered_depth_mask, _ = self.gsoptimizer.renderer([cam],gaussians_mask, scaling_modifier=1.0, bg_color=None)

                mask_image = torch.ones_like(rendered_depth_mask)
                mask_image[rendered_depth_mask==cam.zfar] = 0
                mask_image[rendered_depth<rendered_depth_mask - self.opt['gsoptimizer']['args']['mask_depth_treshold']] = 0
                mask_image = mask_image.squeeze(0).permute(1, 2, 0).repeat(1,1,3).detach().cpu().numpy()
                mask_image = (mask_image.clip(min = 0, max = 1)*255.0).astype(np.uint8)
                mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)

                mask_dilate_size,mask_dilate_iter = self.opt['gsoptimizer']['args']['mask_dilate_size'],self.opt['gsoptimizer']['args']['mask_dilate_iter']
                mask_image_dilate = cv2.dilate(mask_image,np.ones((mask_dilate_size,mask_dilate_size),np.int8),iterations=mask_dilate_iter)

                image_with_mask = ((0.3 * mask_image_dilate + img).clip(min = 0, max = 255) - mask_image).clip(min = 0, max = 255).astype(np.uint8)
  
                input_info_dir = os.path.join(self.out_dir,"input_info")
                if not os.path.exists(input_info_dir):
                    os.makedirs(input_info_dir)

                init_img_and_mask= cv2.hconcat([img, mask_image, mask_image_dilate, image_with_mask])
                cv2.imwrite(f"{input_info_dir}/cam{i+1}_init_img_and_mask.png", init_img_and_mask)

            self.gsoptimizer.train_drag_webui(self,update_view_interval = 10)

    def make_one_camera_pose_frame(self, idx):
        cam = self.colmap_cameras[idx]
        # wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
        # position = -cam.R.T @ cam.T

        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(cam.qvec), cam.T
        ).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = self.server.add_frame(
            f"/colmap/frame_{idx}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=False,
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans


    def render(
        self,
        cam,
        local=False,
        train=False,
        point_tracing=False,
        filter_switch=False,
        filter_gs_switch=False,
        filter_rectangle=None,
    ):
        #self.gaussian.localize = local
        #self.gaussian.alpha = self.alpha_blend_slider.value

        rendered_image, rendered_depth, rendered_alpha, mask_3d = self.gsoptimizer.renderer.render_gui(*self.gaussian,
                                                                                                        edit_mask=self.edit_mask,
                                                                                                        blend_alpha=self.alpha_blend_slider.value,
                                                                                                        viewpoint_camera=cam, 
                                                                                                        scaling_modifier=1.0, 
                                                                                                        bg_color=self.background_tensor,
                                                                                                        save_radii_viewspace_points=False,
                                                                                                        filter_switch=filter_switch,
                                                                                                        filter_rectangle=filter_rectangle)
        
        mask_3d=mask_3d.to(torch.bool)
        if filter_gs_switch:
            self.edit_mask = mask_3d & self.edit_mask

        render_pkg = {}
        render_pkg["comp_rgb"] = rendered_image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        render_pkg["depth"] = rendered_depth.permute(1, 2, 0)[None]
        render_pkg["opacity"] = render_pkg["depth"]/(render_pkg["depth"].max() + 1e-5)

        #ouput point tracing
        render_pkg["point_tracing_start"] = None
        render_pkg["point_tracing_end"] = None
        if point_tracing:
            if self.tracing_point_start_3d is not None:
                #tracing_point_start_2d = project(cam, self.tracing_point_start_3d) # shape [1,2]
                tracing_point_start_2d = cam.project(self.tracing_point_start_3d)
                render_pkg["point_tracing_start"] = tracing_point_start_2d
                if self.tracing_point_end_3d is not None:
                    tracing_point_end_2d = cam.project(self.tracing_point_end_3d)
                    render_pkg["point_tracing_end"] = tracing_point_end_2d

        #output drag point
        render_pkg["handle_points_2d"] = None
        render_pkg["target_points_2d"] = None
        if self.handle_points!=[] and self.target_points!=[] and len(self.handle_points)==len(self.target_points):
            handle_points_2d = cam.project(torch.tensor(self.handle_points).float().to(self.device))
            target_point_2d = cam.project(torch.tensor(self.target_points).float().to(self.device))
            render_pkg["handle_points_2d"] = handle_points_2d
            render_pkg["target_points_2d"] = target_point_2d


        return {
            **render_pkg,
        }


    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None and self.colmap_dir is not None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = CamScene(
                self.colmap_dir, h=-1, w=-1, aspect=self.aspect
            ).cameras
            self.begin_call(list(self.server.get_clients().values())[0])
        viser_cam = list(self.server.get_clients().values())[0].camera
        # viser_cam.up_direction = tf.SO3(viser_cam.wxyz) @ np.array([0.0, -1.0, 0.0])
        # viser_cam.look_at = viser_cam.position
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        # T = viser_cam.position
        if self.render_cameras is None:
            fovy = viser_cam.fov * self.FoV_slider.value
        else:
            fovy = self.render_cameras[0].FoVy * self.FoV_slider.value

        fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
        # fovy = self.render_cameras[0].FoVy
        # fovx = self.render_cameras[0].FoVx
        # math.tan(self.render_cameras[0].FoVx / 2) / math.tan(self.render_cameras[0].FoVy / 2)
        # math.tan(fovx/2) / math.tan(fovy/2)

        # aspect = viser_cam.aspect
        width = int(self.resolution_slider.value)
        height = int(width / self.aspect)
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    def click_cb(self, pointer):
        if self.drag_point_enabled.value and self.add_tracing_point.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos
            click_pos = torch.tensor(click_pos)
            cur_cam = self.camera
            if self.point_tracing_draw_flag:
                self.clear_point_tracing()
                new_value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                self.tracing_point_start_2d.value = new_value
                self.add_tracing_point3d(cur_cam, click_pos)
                print("start point", new_value)
                self.point_tracing_draw_flag = False
            else:
                new_value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                self.tracing_point_end_2d.value = new_value
                self.add_tracing_point3d(cur_cam, click_pos)
                print("end point", new_value)
                self.point_tracing_draw_flag = True

        elif self.draw_filter_bbox.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos
            click_pos = torch.tensor(click_pos)
            cur_cam = self.camera
            if self.draw_flag:
                self.left_up.value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                self.draw_flag = False
            else:
                new_value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                if (self.left_up.value[0] < new_value[0]) and (
                    self.left_up.value[1] < new_value[1]
                ):
                    self.right_down.value = new_value
                    self.draw_flag = True
                else:
                    self.left_up.value = new_value

    def set_system(self, system):
        self.system = system

    def clear_points3d(self):
        self.points3d = []
        self.points3d_neg = []
    
    def clear_point_tracing(self):
        self.tracing_point_start_2d.value = [0, 0]
        self.tracing_point_end_2d.value = [0, 0]
        self.tracing_point_start_3d = None
        self.tracing_point_end_3d = None

    def add_points3d(self, camera, points2d, update_mask=False):
        depth = render(camera, self.gaussian, self.pipe, self.background_tensor)[
            "depth_3dgs"
        ]
        unprojected_points3d = unproject(camera, points2d, depth)
        self.points3d += unprojected_points3d.unbind(0)
        if update_mask:
            self.update_sam_mask_with_point_prompt(self.points3d, self.points3d_neg)
            
    def add_points3d_neg(self, camera, points2d, update_mask=False):
        depth = render(camera, self.gaussian, self.pipe, self.background_tensor)[
            "depth_3dgs"
        ]
        unprojected_points3d = unproject(camera, points2d, depth)
        self.points3d_neg += unprojected_points3d.unbind(0)
        if update_mask:
            self.update_sam_mask_with_point_prompt(self.points3d, self.points3d_neg)
    
    def add_tracing_point3d(self, camera, points2d):
        _, depth, _, _ = self.gsoptimizer.renderer.render_gui(*self.gaussian,
                                                                edit_mask=self.edit_mask,
                                                                viewpoint_camera=camera)
      
        # 把depth中points2d位置的深度替换成位于2D起始点的深度
        
        if self.point_tracing_draw_flag:
            unprojected_points3d = camera.unproject(points2d, depth)
            self.tracing_point_start_3d = unprojected_points3d
        else:
            # use the same detph with start point
            depth[0, int(points2d[1]*camera.image_height), int(points2d[0]*camera.image_width)] = depth[0, self.tracing_point_start_2d.value[1], self.tracing_point_start_2d.value[0]]
            unprojected_points3d = camera.unproject(points2d, depth)
            self.tracing_point_end_3d = unprojected_points3d

    
    @torch.no_grad()
    def prepare_output_image(self, output):
        out_img = output["comp_rgb"][0]  # H W C

        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W


        if (self.drag_point_enabled.value):
            tracing_start, tracing_end = output["point_tracing_start"], output["point_tracing_end"]
            if tracing_start is not None:
                if tracing_end is not None:
                    dtype = out_img.dtype
                    device = out_img.device
                    
                    out_img_np =  out_img.cpu().permute(1,2,0).numpy()
                    out_img_np = cv2.cvtColor(out_img_np, cv2.COLOR_RGB2BGR)

                    cv2.circle(out_img_np, (int(tracing_start[0,0]), int(tracing_start[0,1])), 12, (255, 255, 255), -1)
                    cv2.circle(out_img_np, (int(tracing_end[0,0]), int(tracing_end[0,1])), 12, (255, 255, 255), -1)
                    out_img = cv2.arrowedLine(out_img_np, (int(tracing_start[0,0]), int(tracing_start[0,1])),(int(tracing_end[0,0]), int(tracing_end[0,1])), (255, 255, 255), 3, tipLength=0.5)
          
                    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                    out_img = torch.from_numpy(out_img).permute(2, 0, 1).to(dtype).to(device)
                else:
                    out_img = torchvision.utils.draw_keypoints(
                        out_img,
                        tracing_start[None, ...],
                        colors="white",
                        radius=12,
                    )

        if output["handle_points_2d"] is not None and output["target_points_2d"] is not None:
            handle_points_2d, target_points_2d = output["handle_points_2d"], output["target_points_2d"]
            if handle_points_2d.shape == target_points_2d.shape:
                dtype = out_img.dtype
                device = out_img.device
                out_img_np =  out_img.cpu().permute(1,2,0).numpy()
                out_img_np = cv2.cvtColor(out_img_np, cv2.COLOR_RGB2BGR)
                for i in range(handle_points_2d.shape[0]):
                    cv2.circle(out_img_np, (int(handle_points_2d[i,0]), int(handle_points_2d[i,1])), 12, (255, 0, 0), -1)
                    cv2.circle(out_img_np, (int(target_points_2d[i,0]), int(target_points_2d[i,1])), 12, (0, 0, 255), -1)
                    out_img = cv2.arrowedLine(out_img_np, (int(handle_points_2d[i,0]), int(handle_points_2d[i,1])),(int(target_points_2d[i,0]), int(target_points_2d[i,1])), (255, 255, 255), 3, tipLength=0.5)
                
                out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                out_img = torch.from_numpy(out_img).permute(2, 0, 1).to(dtype).to(device)  

        local_left_up = self.left_up.value
        locals_right_down = self.right_down.value
        if self.draw_filter_bbox.value and self.draw_flag and (local_left_up[0] < locals_right_down[0]) and (local_left_up[1] < locals_right_down[1]):
            # 获取目标区域的原始颜色
            region = out_img[
                :,
                self.left_up.value[1] : self.right_down.value[1],
                self.left_up.value[0] : self.right_down.value[0],
            ]

            # 将目标区域的颜色加上一些白色
            # 这里假设白色增加的比例为0.2，可以根据需要调整
            render_alpha = 0.6
            out_img[
                :,
                self.left_up.value[1] : self.right_down.value[1],
                self.left_up.value[0] : self.right_down.value[0],
            ] = render_alpha * region + (1 - render_alpha) * 255
        
        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    def render_loop(self):
        while True:
            # if self.viewer_need_update:
            self.update_viewer()
            time.sleep(1e-4)

    @torch.no_grad()
    def update_viewer(self):
        gs_camera = self.camera
        if gs_camera is None:
            return
        
        filter_gs_switch = False
        if self.filter_gs_switch:
            # 过滤对应高斯
            filter_gs_switch = True
            self.filter_gs_switch = False
            
        filter_rectangle = None
        if (
            self.filter_enabled.value
            and self.draw_filter_bbox.value
            and self.draw_flag
            and (self.left_up.value[0] < self.right_down.value[0])
            and (self.left_up.value[1] < self.right_down.value[1])
        ):
            filter_rectangle = torch.tensor([self.left_up.value[0],self.left_up.value[1], self.right_down.value[0], self.right_down.value[1]]).to(torch.float32).to(self.device)
        output = self.render(gs_camera, point_tracing=self.drag_point_enabled.value,
                             filter_switch=self.filter_enabled.value, filter_gs_switch=filter_gs_switch, filter_rectangle=filter_rectangle)

        out = self.prepare_output_image(output)
        self.server.set_background_image(out, format="jpeg")


    @torch.no_grad()
    def render_cameras_list(self, edit_cameras):
        origin_frames = []
        for cam in edit_cameras:
            out = self.render(cam)["comp_rgb"]
            origin_frames.append(out)

        return origin_frames


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #

    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--use_triplane", default=True)
    parser.add_argument("--base_dir", default='logs/')
    parser.add_argument("--output_dir", default='./exps/tmp')

    parser.add_argument("--gpu", type=int, default=0)

    args, extras = parser.parse_known_args()
    args.out_dir = os.path.join(args.base_dir, args.output_dir)
    
    print(args)
    opt = OmegaConf.load(args.config)
    
    #args = parser.parse_args()
    webui = WebUI(args,opt)
    webui.render_loop()
