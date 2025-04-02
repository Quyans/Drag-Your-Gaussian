import os
import torch
import random
import numpy as np
import imageio
import tqdm
from utils import sample_from_dense_cameras, export_ply_for_gaussians, import_str, matrix_to_square, export_video
from PIL import Image
import torch.nn.functional as F
from modules.optimizers.gs_optimizer import GSOptimizer
from modules.optimizers.gs_utils import load_ply
from modules.scene.camera_scene import CamScene

from torchvision.utils import save_image
import shutil
from utils import seed_everything

import argparse
from omegaconf import OmegaConf
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    #output dir
    parser.add_argument("--base_dir", default='logs/')
    parser.add_argument("--output_dir", default='./exps/tmp')
    #input
    parser.add_argument("--gs_source", default=None)
    parser.add_argument("--colmap_dir", default=None)
    parser.add_argument("--point_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    
    parser.add_argument("--gpu", type=int, default=0)
    
    parser.add_argument("--from_director3d", action='store_true', default=False)

    args, extras = parser.parse_known_args()
    args.out_dir = os.path.join(args.base_dir, args.output_dir)
    
    print(args)

    opt = OmegaConf.load(args.config)
    #if input from the command line
    if args.gs_source is not None:
        opt['scene']['gs_source'] = args.gs_source
    if args.colmap_dir is not None:
        opt['scene']['colmap_dir'] = args.colmap_dir
    if args.point_dir is not None:
        with open(args.point_dir, "r") as file:
            data = json.load(file)
        opt['scene']['handle_points'] = data.get("handle_points", [])
        opt['scene']['target_points'] = data.get("target_points", [])
    if args.mask_dir is not None:
        opt['scene']['mask_dir'] = args.mask_dir

    torch.backends.cudnn.benchmark = True

    device = f'cuda:{args.gpu}'
    
    # move config files to out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(args.out_dir, 'config.yaml'))

    seed_everything(0)

    if args.from_director3d:
        camera_npy_files = [f for f in os.listdir(opt['scene']['colmap_dir']) if f.endswith('.npy')]
        assert len(camera_npy_files) == 1
        #B N K
        camera_npy = np.load(os.path.join(opt['scene']['colmap_dir'], camera_npy_files[0]))
        camera_npy = torch.tensor(camera_npy,device=device)[0]
        cameras_extent = 2
        colmap_cameras = []
        for i in range(camera_npy.shape[0]):
            from utils import convert_camera_parameters_into_viewpoint_cameras
            colmap_cameras.append(convert_camera_parameters_into_viewpoint_cameras(camera_npy[i],image_name=f'frame_{i:05}',render_res_factor=opt['scene']['render_res_factor']))
    else:
        scene = CamScene(opt['scene']['colmap_dir'], h=-1, w=-1, render_res_factor=opt['scene']['render_res_factor'])
        cameras_extent = scene.cameras_extent
        colmap_cameras = scene.cameras

    edit_mask = torch.load(opt['scene']['mask_dir']).cuda()
    gsoptimizer = GSOptimizer(**opt['gsoptimizer']['args'],
                              image_height=colmap_cameras[0].image_height//colmap_cameras[0].render_res_factor,
                              image_width=colmap_cameras[0].image_width//colmap_cameras[0].render_res_factor,
                              train_args=args,
                              ).to(device)
    
    gaussians = load_ply(opt['scene']['gs_source'])
    xyz, features, opacity, scales, rotations = gaussians
    gaussians_mask = (xyz[edit_mask],features[edit_mask],opacity[edit_mask],scales[edit_mask],rotations[edit_mask])

    handle_points = opt['scene']['handle_points']
    target_points = opt['scene']['target_points']

    #delete invaild camera pose
    train_colmap_cameras = []
    for i,cam in enumerate(colmap_cameras):
        #rendered_image, rendered_depth, rendered_mask = refiner.renderer.render(*gaussians, cam, scaling_modifier=1.0, bg_color=None)
        rendered_image, rendered_depth, rendered_mask = gsoptimizer.renderer([cam],gaussians, scaling_modifier=1.0, bg_color=None)

        assert len(handle_points) == len(target_points)
        vaild_flag = True
        for j in range(len(handle_points)):
            h_pixel,h_depth = cam.world_to_pixel(handle_points[j])
            g_pixel,g_depth = cam.world_to_pixel(target_points[j])
            if h_depth is None or g_depth is None: 
                # or h_depth > rendered_depth[0,0,int(h_pixel[1]),int(h_pixel[0])] + 1 or \
                # g_depth > rendered_depth[0,0,int(g_pixel[1]),int(g_pixel[0])] + 1:
                vaild_flag = False
        
        rendered_image, rendered_depth, rendered_mask = gsoptimizer.renderer([cam],gaussians, scaling_modifier=1.0, bg_color=None)
        rendered_image_mask, rendered_depth_mask, _ = gsoptimizer.renderer([cam],gaussians_mask, scaling_modifier=1.0, bg_color=None)
        mask_image = torch.ones_like(rendered_depth_mask)
        mask_image[rendered_depth_mask==cam.zfar] = 0
        mask_image[rendered_depth<rendered_depth_mask - opt['gsoptimizer']['args']['mask_depth_treshold']] = 0
        if (mask_image==1).sum() < 50:
            vaild_flag = False

        if vaild_flag:
            train_colmap_cameras.append(cam)

    #valid camera number must > 10
    assert len(colmap_cameras) > 10

    #output init image and point
    for i,cam in enumerate(train_colmap_cameras):
        #rendered_image, rendered_depth, rendered_mask = refiner.renderer.render(*gaussians, cam, scaling_modifier=1.0, bg_color=None)
        rendered_image, rendered_depth, rendered_mask = gsoptimizer.renderer([cam],gaussians, scaling_modifier=1.0, bg_color=None)
        
        import cv2
        img = (rendered_image/2 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = (img.clip(min = 0, max = 1)*255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        assert len(handle_points) == len(target_points)
        for j in range(len(handle_points)):
            h_pixel,_ = cam.world_to_pixel(handle_points[j])
            g_pixel,_ = cam.world_to_pixel(target_points[j])
            cv2.circle(img, (int(h_pixel[0]),int(h_pixel[1])), 8, (255, 0, 0), -1)
            cv2.circle(img, (int(g_pixel[0]),int(g_pixel[1])), 8, (0, 0, 255), -1)
            cv2.arrowedLine(img, (int(h_pixel[0]),int(h_pixel[1])), 
                                 (int(g_pixel[0]),int(g_pixel[1])), (255, 255, 255), 2, tipLength=0.5)
            # color = (0,0,255) if j == 0 else ((0,255,0) if j == 1 else (255,0,0))
            # cv2.arrowedLine(img, (int(h_pixel[0]),int(h_pixel[1])), 
            #                     (int(g_pixel[0]),int(g_pixel[1])), color)
        init_dir = os.path.join(args.out_dir,"init")
        if not os.path.exists(init_dir):
            os.makedirs(init_dir)
        cv2.imwrite(f"{init_dir}/cam{i+1}_{cam.image_name}.png",img)

        rendered_image_mask, rendered_depth_mask, _ = gsoptimizer.renderer([cam],gaussians_mask, scaling_modifier=1.0, bg_color=None)

        mask_image = torch.ones_like(rendered_depth_mask)
        mask_image[rendered_depth_mask==cam.zfar] = 0
        mask_image[rendered_depth<rendered_depth_mask - opt['gsoptimizer']['args']['mask_depth_treshold']] = 0
        mask_image = mask_image.squeeze(0).permute(1, 2, 0).repeat(1,1,3).detach().cpu().numpy()
        mask_image = (mask_image.clip(min = 0, max = 1)*255.0).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)

        #cv2.imwrite(f"{init_dir}/cam{i+1}_mask.png",mask_image)

        mask_dilate_size, mask_dilate_iter = opt['gsoptimizer']['args']['mask_dilate_size'], opt['gsoptimizer']['args']['mask_dilate_iter']
        mask_image_dilate = cv2.dilate(mask_image,np.ones((mask_dilate_size,mask_dilate_size),np.int8),iterations=mask_dilate_iter)
        #cv2.imwrite(f"{init_dir}/cam{i+1}_mask_dilate.png",mask_image_dilate)

        image_with_mask = ((0.3 * mask_image_dilate + img).clip(min = 0, max = 255) - mask_image).clip(min = 0, max = 255).astype(np.uint8)
        #cv2.imwrite(f"{init_dir}/cam{i+1}_image_with_mask.png",image_with_mask)
        input_info_dir = os.path.join(args.out_dir,"input_info")
        if not os.path.exists(input_info_dir):
            os.makedirs(input_info_dir)
        init_img_and_mask= cv2.hconcat([img, mask_image, mask_image_dilate, image_with_mask])
        cv2.imwrite(f"{input_info_dir}/cam{i+1}_init_img_and_mask.png", init_img_and_mask)

    gsoptimizer.train_drag(gaussians,train_colmap_cameras,cameras_extent,edit_mask,handle_points,target_points)

    #export_ply_for_gaussians(os.path.join(args.out_dir, 'ply', f'{filename}{extra_filename}'), result['gaussians'])

    