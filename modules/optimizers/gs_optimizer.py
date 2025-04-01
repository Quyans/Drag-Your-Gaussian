import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from models.ip_adapter import ImageProjModel
from models.appearance_encoder import AppearanceEncoderModel
from models.point_embedding import PointEmbeddingModel

from modules.renderers.gaussians_renderer import GaussianRenderer

from utils import sample_from_dense_cameras, inverse_sigmoid, export_ply_for_gaussians, export_points_for_gaussians
import time
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import copy
import random
from safetensors.torch import load_file

from .gs_utils import GaussiansManager
from accelerate import Accelerator
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from drag_pipeline import DragPipeline
from diffusers.training_utils import unet_lora_state_dict
from drag_util import  import_model_class_from_model_name_or_path
from diffusers.optimization import get_scheduler
from torchvision import transforms
from modules.optimizers.hexplane import HexPlaneField
from modules.utils.video_utils import create_video_from_images, create_video_from_two_folders

class GSOptimizer(nn.Module):
    def __init__(self, 
            base_sd_path='checkpoints/stable-diffusion-inpainting',
            vae_path='checkpoints/sd-vae-ft-mse/',
            ip_adapter_path='checkpoints/IP-Adapter/models/',
            lightning_drag_model_path='checkpoints/lightning-drag-sd15',
            lcm_lora_path=None,
            total_iterations=1000,
            triplane_optim_iter=-1,
            triplane_optim_step_percent=0.7,
            triplane_lr_down_scale=0.1,
            batch_size = 2,
            guidance_scale=7.5,
            min_step_percent=0.02, 
            max_step_percent=0.75,
            lr_scale=1,
            lr_scale_end=1,
            lrs={'xyz': 0.0001, 'features': 0.01, 'opacity': 0.05, 'scales': 0.01, 'rotations': 0.01}, 
            lambda_latent_sds=1,
            lambda_image_sds=0.01,
            lambda_reg=1,
            use_pos_deform2=True,
            mask_depth_treshold=0.1,
            mask_dilate_size=30,
            mask_dilate_iter=2,
            only_optimize_3d_mask=True,
            use_knn=False,
            knn_coeff=0.3,
            other_coeff=0,
            knn_number=5,
            num_densifications=5,
            image_height=512,
            image_width=512,
            train_args=None,
        ):
        super().__init__()
        self.device = "cuda"
        self.dtype = torch.float16
        self.output_dir = train_args.out_dir
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_sd_path,
            subfolder="tokenizer",
            use_fast=False,
        )
        # Load text encoder
        text_encoder_cls = import_model_class_from_model_name_or_path(base_sd_path, revision=None)
        text_encoder = text_encoder_cls.from_pretrained(
            base_sd_path, subfolder="text_encoder"
        ).requires_grad_(False)
        # Load vae
        vae = AutoencoderKL.from_pretrained(vae_path).requires_grad_(False)
        # Load inpaint unet
        unet = UNet2DConditionModel.from_pretrained(
            base_sd_path, subfolder="unet"
        ).requires_grad_(False)

        scheduler = DDIMScheduler.from_pretrained(base_sd_path,
                        subfolder="scheduler")
        
        config = unet.config
        config['in_channels'] = 4
        appearance_encoder = AppearanceEncoderModel.from_config(config)

        # Load image encoder
        image_encoder_path = os.path.join(ip_adapter_path, "image_encoder")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        clip_image_processor = CLIPImageProcessor()
        # Load ip-adapter image proj model
        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=image_encoder.config.projection_dim,
            clip_extra_context_tokens=4, # HACK: hard coded to be 4 here, as we are using the normal ip adapter
        )
        ip_ckpt_path = os.path.join(ip_adapter_path, "ip-adapter_sd15.bin")
        ip_state_dict = torch.load(ip_ckpt_path, map_location="cpu", weights_only=True)
        image_proj_model.load_state_dict(ip_state_dict["image_proj"])

        point_embedding = PointEmbeddingModel(embed_dim=16)
        self.pipe = DragPipeline(
            vae=vae.to(self.device).to(self.dtype),
            text_encoder=text_encoder.to(self.device).to(self.dtype),
            tokenizer=tokenizer,
            unet=unet.to(self.device).to(self.dtype),
            appearance_encoder=appearance_encoder.to(self.device).to(self.dtype),
            scheduler=scheduler,
            feature_extractor=clip_image_processor,
            image_encoder=image_encoder.to(self.device).to(self.dtype),
            point_embedding=point_embedding.to(self.device).to(self.dtype),
            safety_checker=None,
            fusion_blocks="full",
            initialize_attn_processor=True,
            use_norm_attn_processor=True,
            initialize_ip_attn_processor=True,
            image_proj_model=image_proj_model.to(self.device).to(self.dtype),
        )

        self.load_model(lightning_drag_model_path, base_sd_path, ip_state_dict)
        
        self.lcm_lora_path = lcm_lora_path
        if self.lcm_lora_path is not None:
            self.pipe.load_lora_weights(lcm_lora_path)
            self.pipe.fuse_lora()
            self.pipe.scheduler = LCMScheduler.from_pretrained(base_sd_path, subfolder="scheduler")

        self.pipe = self.pipe.to(self.device).to(self.dtype)

        self.total_iterations = total_iterations
        if triplane_optim_iter == -1:
            self.triplane_optim_iter = int((self.total_iterations/torch.pi)*torch.arccos(torch.tensor(2*(triplane_optim_step_percent-min_step_percent)/(max_step_percent-min_step_percent)-1)))
        else:
            self.triplane_optim_iter = triplane_optim_iter
        self.triplane_lr_down_scale = triplane_lr_down_scale

        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.lrs = {key: value * lr_scale for key, value in lrs.items()}

        # grid net
        self.create_grid_net()
        self.create_grid_net_optimizer()
        
        self.lr_scale = lr_scale
        self.lr_scale_end = lr_scale_end

        self.register_buffer("alphas_cumprod", self.pipe.scheduler.alphas_cumprod, persistent=False)

        self.device = 'cpu'

        self.num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps

        self.set_min_max_steps(min_step_percent, max_step_percent)

        self.renderer = GaussianRenderer()

        self.lambda_latent_sds = lambda_latent_sds
        self.lambda_image_sds = lambda_image_sds
        self.lambda_reg = lambda_reg
        self.use_pos_deform2 = use_pos_deform2

        self.mask_dilate_size = mask_dilate_size
        self.mask_dilate_iter = mask_dilate_iter

        self.only_optimize_3d_mask = only_optimize_3d_mask
        
        self.use_knn = use_knn
        self.knn_coeff = knn_coeff
        self.other_coeff = other_coeff
        self.knn_number = knn_number

        self.opacity_threshold = 0.01
        self.densification_interval = self.total_iterations // (num_densifications + 1)
        
        #downsample factor 2
        self.image_height = int(image_height)
        self.image_width = int(image_width)

        self.mask_depth_treshold = mask_depth_treshold

    # create net for grid feature
    def create_grid_net(self):
        kplanes_config = {
                            'grid_dimensions': 2,
                            'input_coordinate_dim': 3,
                            'output_coordinate_dim': 32,
                            'resolution': [64, 64, 64]
                        }
        multires = [1, 2, 4, 8]
        bounds = 1.6
        self.grid = HexPlaneField(bounds, kplanes_config, multires)
        
        grid_out_dim = self.grid.feat_dim
        # grid_out_dim = 3
        linear_out = 64
        
        self.feature_out = [nn.Linear(grid_out_dim ,linear_out)]
        
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(linear_out,linear_out),nn.ReLU(),nn.Linear(linear_out, 3))
        
        self.pos_deform_2 = nn.Sequential(nn.ReLU(),nn.Linear(linear_out,linear_out),nn.ReLU(),nn.Linear(linear_out, 3))
        
        # self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(linear_out,linear_out),nn.ReLU(),nn.Linear(linear_out, 3))
        #self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(linear_out,linear_out),nn.ReLU(),nn.Linear(linear_out, 4))
                
        # 初始化最后一层的权重和偏置为0
        nn.init.constant_(self.pos_deform[-1].weight, 0)
        nn.init.constant_(self.pos_deform[-1].bias, 0)

        nn.init.constant_(self.pos_deform_2[-1].weight, 0)
        nn.init.constant_(self.pos_deform_2[-1].bias, 0)
       
    def create_grid_net_optimizer(self):
        self.grid_net_optimizer = torch.optim.Adam([
            {'params': self.grid.parameters(), 'lr': 1e-3},
            {'params': self.feature_out.parameters(), 'lr': 5e-4},
            {'params': self.pos_deform.parameters(), 'lr': 5e-4},
            {'params': self.pos_deform_2.parameters(), 'lr': 5e-4},
            # {'params': self.scales_deform.parameters(), 'lr': 5e-4},
            # {'params': self.rotations_deform.parameters(), 'lr': 5e-4}
        ])
        

    def load_model(self, lightning_drag_path, base_sd_path, ip_state_dict):

        # Load weights for attn_processors, including those from IP-Adapter
        attn_processors = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        state_dict = torch.load(
            os.path.join(lightning_drag_path, "lightning-drag-sd15-attn.bin")
        )
        state_dict.update(ip_state_dict["ip_adapter"])
        attn_processors.load_state_dict(state_dict)

        # Load appearance encoder
        appearance_state_dict = load_file(
            os.path.join(lightning_drag_path,
                         "appearance_encoder/diffusion_pytorch_model.safetensors")
        )
        self.pipe.appearance_encoder.load_state_dict(appearance_state_dict)

        # Load point embedding
        point_embedding_state_dict = torch.load(
            os.path.join(lightning_drag_path, "point_embedding/point_embedding.pt")
        )
        self.pipe.point_embedding.load_state_dict(point_embedding_state_dict)

    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
    
    def to(self, device):
        self.device = device
        return super().to(device)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation_strategy='longest_first',
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(inputs.input_ids.to(next(self.text_encoder.parameters()).device))[0]
        return text_embeddings
    
    # @torch.cuda.amp.autocast(enabled=False)
    def encode_image(self, images):
        images = 2 * images - 1 
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    # @torch.cuda.amp.autocast(enabled=False)
    def decode_latent(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        return images
    
    def predict_delta_xyz(self, xyz_normalized,edit_mask):
        grid_feature = self.grid(xyz_normalized)
        feature_out = self.feature_out(grid_feature)
        delta_xyz_1 = self.pos_deform(feature_out)
        delta_xyz_2 = self.pos_deform_2(feature_out[~edit_mask].detach())
        delta_xyz = delta_xyz_1
        if self.use_pos_deform2:
            delta_xyz[~edit_mask] = delta_xyz[~edit_mask].detach() + delta_xyz_2

        return delta_xyz
    
    
    def predict_delta_xyz_scale(self, xyz_normalized,edit_mask):
        grid_feature = self.grid(xyz_normalized)
        feature_out = self.feature_out(grid_feature)
        delta_xyz_1 = self.pos_deform(feature_out)
        delta_xyz_2 = self.pos_deform_2(feature_out[~edit_mask].detach())
        delta_xyz = delta_xyz_1
        if self.use_pos_deform2:
            delta_xyz[~edit_mask] = delta_xyz[~edit_mask] + delta_xyz_2
            
        delta_scale = self.scales_deform(feature_out)
        delta_scale[~edit_mask] = 0.0

        return delta_xyz, delta_scale
    
    def predict_delta_xyz_scale_rotation(self, xyz_normalized):
        grid_feature = self.grid(xyz_normalized)
        feature_out = self.feature_out(grid_feature)
        delta_xyz = self.pos_deform(feature_out)
        delta_scale = self.scales_deform(feature_out)
        delta_rotation = self.rotations_deform(feature_out)
        return delta_xyz, delta_scale, delta_rotation
    
    def train_drag(self, gaussians, colmap_cameras, camera_extent, edit_mask, handle_points, target_points):
        batch_size = self.batch_size
        gaussians_init = copy.deepcopy(gaussians)

        xyz, features, opacity, scales, rotations = gaussians_init
        gs_init_mask = (xyz[edit_mask],features[edit_mask],opacity[edit_mask],scales[edit_mask],rotations[edit_mask])

        xyz, features, opacity, scales, rotations = gaussians
        gs_mask = opacity[..., 0] >= self.opacity_threshold

        xyz_original = xyz[gs_mask]
        features_original = features[gs_mask]
        opacity_original = opacity[gs_mask]
        scales_original = scales[gs_mask]
        rotations_original = rotations[gs_mask]

        edit_mask = edit_mask[gs_mask]
        gaussians_optim = GaussiansManager(xyz_original, features_original, opacity_original, scales_original, rotations_original, self.lrs, 
                                           edit_mask,self.only_optimize_3d_mask,
                                           use_knn=self.use_knn,knn_coeff=self.knn_coeff,other_coeff=self.other_coeff,knn_number=self.knn_number)

        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gaussians_optim.optimizer, gamma=(self.lr_scale_end / self.lr_scale) ** (1 / self.total_iterations))
        #lr_scheduler_triplane = torch.optim.lr_scheduler.ExponentialLR(self.grid_net_optimizer, gamma=(0.01 / 1) ** (1 / self.total_iterations))

        self.pipe.prepare_embeddings(gaussians,colmap_cameras,
                                     handle_points,target_points,
                                     height=self.image_height,
                                     width=self.image_width,)


        #region NOTE point traj
        # xyz_list = []
        # sample_point_traj_num = 200
        # from point_traj.utils import farthest_point_sampling,sample_points_with_mask,sample_from_masks,generate_distinct_colors,filter_sample_points
        # #point_traj_mask = torch.load("./point_traj_mask.pt").cuda()[gs_mask]
        # #mask_intersection = point_traj_mask & edit_mask
        # #sample_indice = sample_from_masks(point_traj_mask,edit_mask,sample_point_traj_num)
        # xyz, features, opacity, scales, rotations = gaussians_optim()
        # sample_indice = farthest_point_sampling(xyz[gaussians_optim.get_3dmask()],sample_point_traj_num)
        # sample_indice = filter_sample_points(xyz, sample_indice.cuda(), 0.2, 10)
        # print("sample_point_num:"+str(len(sample_indice)))

        # #sample_color = generate_distinct_colors(sample_point_traj_num)
        # point_color = [(236,52,73),(238,130,85),(226,153,63),(245,239,15),(141,210,22),(28,192,60),(48,214,159),(26,171,214),(32,62,205),(103,49,147),(126,52,159),(210,46,223),(226,63,205)]
        # y_coords = xyz[sample_indice][:,1]
        # y_min, y_max = y_coords.min().item(), y_coords.max().item()
        # interval_bounds = torch.linspace(y_min, y_max, len(point_color) + 1).cuda()
        # point_color_indice = torch.bucketize(y_coords, interval_bounds) - 1

        # #sample other
        # other_sample_point_traj_num = 2000
        # other_mask = gaussians_optim.get_othermask()
        # other_sample_indice = farthest_point_sampling(xyz[other_mask],other_sample_point_traj_num).cuda()
        # other_sample_indice = other_sample_indice + (gaussians_optim.masks_lens_group[0] + gaussians_optim.masks_lens_group[1])
        
        # sample_indice = torch.cat([sample_indice,other_sample_indice],dim=0)
        
        # point_color.append((174,171,171))
        # other_colo_indice = torch.zeros_like(other_sample_indice).cuda()
        # other_colo_indice[:] = len(point_color) - 1 
        
        # point_color_indice = torch.cat([point_color_indice,other_colo_indice],dim=0)
        
        # #assert torch.all(edit_mask[sample_indice])
        # #assert torch.all(point_traj_mask[sample_indice])
        # xyz_list.append(xyz.clone())
        #endregion
        
        with torch.no_grad():
            for i,cam in enumerate(colmap_cameras):  
                xyz, features, opacity, scales, rotations = gaussians_optim()
                gs_optim = (xyz, features, opacity, scales, rotations)
                rendered_image_optim, rendered_depth, rendered_mask = self.renderer([cam],gs_optim, scaling_modifier=1.0, bg_color=None,save_radii_viewspace_points=False,down_sample_res=False)
                import cv2


        for n in tqdm(range(self.total_iterations),desc="trainning drag"):
            xyz, features, opacity, scales, rotations = gaussians_optim()
            xyz_normalized = gaussians_optim.normalize_xyz(xyz.detach())
            
            if self.use_knn:
                delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
            else:
                delta_xyz = self.predict_delta_xyz(xyz_normalized,gaussians_optim.get_3dmask())
            
            #NOTE ablation triplane
            gs_optim = (xyz + delta_xyz, features, opacity, scales, rotations)

            #sample camera
            camera_idx_list = random.sample(range(len(colmap_cameras)), batch_size)
            camera_list = [colmap_cameras[i] for i in camera_idx_list]
     
            #[B,N,2]
            handle_points_pixel_list = []
            target_points_pixel_list = []
            for camera in camera_list:
                #camera = colmap_cameras[i]
                h_pixel = []
                g_pixel = []
                for j in range(len(handle_points)):
                    #swap tensor[0] and tensor[1]
                    #downsample factor 2
                    h_pixel_xy,_ = camera.world_to_pixel(handle_points[j])
                    g_pixel_xy,_ = camera.world_to_pixel(target_points[j])
                    #xy space to hw space
                    h_pixel.append(h_pixel_xy[[1, 0]].unsqueeze(0))
                    g_pixel.append(g_pixel_xy[[1, 0]].unsqueeze(0))
                handle_points_pixel_list.append(torch.cat(h_pixel,dim=0).long())
                target_points_pixel_list.append(torch.cat(g_pixel,dim=0).long())
            
            rendered_image_init, rendered_depth_init, _ = self.renderer(camera_list,gaussians_init)
            rendered_image_optim, rendered_depth_optim, _ = self.renderer(camera_list,gs_optim)
            
            #rendered_depth for mask
            with torch.no_grad():
                _, rendered_depth_mask, _ = self.renderer(camera_list,gs_init_mask,save_radii_viewspace_points=False)
                mask_image = torch.ones_like(rendered_depth_mask)
                mask_image[rendered_depth_mask==camera_list[0].zfar] = 0
                mask_image[rendered_depth_init<rendered_depth_mask-self.mask_depth_treshold] = 0
                mask_image = [mask_image[i].squeeze(0).cpu().numpy().astype(np.uint8) for i in range(mask_image.shape[0])]
                #dilate mask
                #mask_image_dilate = cv2.dilate(mask_image,np.ones((25,25),np.int8),iterations=1)
                import cv2
                drag_mask = [Image.fromarray(cv2.dilate(mask * 255,np.ones((self.mask_dilate_size,self.mask_dilate_size),np.int8),iterations=self.mask_dilate_iter)) for mask in mask_image]
                #drag_mask = np.ones((self.image_height, self.image_width))
                #drag_mask = [Image.fromarray(drag_mask * 255)] * batch_size

            step_ratio = n / self.total_iterations
            #t = torch.full((1,), int(step_ratio ** (1/2) * (self.min_step - self.max_step) + self.max_step), dtype=torch.long, device=self.device)
            #t = torch.full((1,), int(0.5 * (self.max_step - self.min_step)*(1+torch.cos(torch.pi*torch.tensor(step_ratio))) + self.min_step), dtype=torch.long, device=self.device)
            #t = torch.randint(self.min_step, self.max_step, (1,), dtype=torch.long, device=self.device)
            t = torch.full((1,), int(step_ratio * (self.min_step - self.max_step) + self.max_step), dtype=torch.long, device=self.device)
            
            cur_guidance_scale_points = (self.guidance_scale-1.0) * (1.0 - step_ratio) ** 2 + 1.0
            #cur_guidance_scale_points = self.guidance_scale


            loss_latent_sds, loss_image_sds, loss_embedding_lora = \
                self.pipe(
                    ref_image=rendered_image_init,
                    render_image=rendered_image_optim,
                    mask_image=drag_mask,
                    handle_points_pixel_list=handle_points_pixel_list,
                    target_points_pixel_list=target_points_pixel_list,
                    time_step=t,
                    guidance_scale_points=cur_guidance_scale_points,
                    prompt=[""] * batch_size,
                    height=self.image_height,
                    width=self.image_width,
                    num_train_timesteps=self.num_train_timesteps,
                )
            loss_reg = F.l1_loss(delta_xyz[gaussians_optim.get_knnmask()], torch.zeros_like(delta_xyz[gaussians_optim.get_knnmask()]), reduction="mean")            
            loss = loss_latent_sds * self.lambda_latent_sds + loss_image_sds * self.lambda_image_sds + loss_embedding_lora + loss_reg * self.lambda_reg

            loss.backward()
            
            #NOTE Ablation
            if n > self.triplane_optim_iter:
                gaussians_optim.optimizer.step()
            gaussians_optim.optimizer.zero_grad()

            self.grid_net_optimizer.step()
            self.grid_net_optimizer.zero_grad()
            #NOTE Ablation
            if n == self.triplane_optim_iter:
                for param_group in self.grid_net_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.triplane_lr_down_scale
            
            self.pipe.unet_emb_lora_optimizer.step()
            self.pipe.unet_emb_lora_optimizer.zero_grad()
            #lr_scheduler_triplane.step()

            if n > self.triplane_optim_iter and step_ratio > 0.25 and step_ratio <= 0.8:
                for radii, viewspace_points in zip(self.renderer.radii, self.renderer.viewspace_points):
                    visibility_filter = radii > 0
                    gaussians_optim.is_visible[visibility_filter] = 1
                    gaussians_optim.max_radii2D[visibility_filter] = torch.max(gaussians_optim.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians_optim.add_densification_stats(viewspace_points, visibility_filter)
                if n % self.densification_interval == 0 and n != 0:
                    gaussians_optim.densify_and_prune(max_grad=10,extent=camera_extent,opacity_threshold=0.01)
            
            with torch.no_grad():
                if (n % 500==0 and n!=0) or n == self.total_iterations - 1 or n==self.triplane_optim_iter:
                    xyz, features, opacity, scales, rotations = gaussians_optim()
                    xyz_normalized = gaussians_optim.normalize_xyz(xyz.detach())
                    # delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
                    if self.use_knn:
                        delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
                    else:
                        delta_xyz = self.predict_delta_xyz(xyz_normalized,gaussians_optim.get_3dmask())
                    gs_optim = (xyz + delta_xyz, features, opacity, scales, rotations)

                    for i,cam in enumerate(colmap_cameras):
                        rendered_image_optim, rendered_depth, rendered_mask = self.renderer([cam],gs_optim, scaling_modifier=1.0, bg_color=None,save_radii_viewspace_points=False)
                        import cv2
                        img = (rendered_image_optim/2 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                        img = (img.clip(min = 0, max = 1)*255.0).astype(np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        #draw arrowline
                        # assert len(handle_points) == len(target_points)
                        # for j in range(len(handle_points)):
                        #     h_pixel,_ = cam.world_to_pixel(handle_points[j])
                        #     g_pixel,_ = cam.world_to_pixel(target_points[j])
                        #     color = (0,0,255) if j == 0 else ((0,255,0) if j == 1 else (255,0,0))
                        #     cv2.arrowedLine(img, (int(h_pixel[0]),int(h_pixel[1])), (int(g_pixel[0]),int(g_pixel[1])), color)

                        if not os.path.exists(f"{self.output_dir}/optim_{n}/"):
                            os.makedirs(f"{self.output_dir}/optim_{n}/")
                        cv2.imwrite(f"{self.output_dir}/optim_{n}/cam_{i+1}_{cam.image_name}.png",img)

                        #region output point traj
                        # if len(xyz_list)>1:
                        #     img_res = cv2.imread(f'./logs/point_traj/render_1024_1024/cam{i+1}.png')
                        #     rendered_image_optim, rendered_depth, rendered_mask = self.renderer([cam],gs_optim, scaling_modifier=1.0, bg_color=None,save_radii_viewspace_points=False,down_sample_res=False)
                        
                        #     img = (rendered_image_optim/2 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                        #     img = (img.clip(min = 0, max = 1)*255.0).astype(np.uint8)
                        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        #     #point_traj_xyz = gs_optim[0][sample_indice].tolist()
                        #     # for jj in range(len(point_traj_xyz)):
                        #     #     point_traj_pixel,point_traj_depth = cam.world_to_pixel(point_traj_xyz[jj])
                        #     #     if point_traj_pixel is not None and point_traj_depth < rendered_depth[0,0,int(point_traj_pixel[1]),int(point_traj_pixel[0])]:
                        #     #         cv2.circle(img, (int(point_traj_pixel[0]),int(point_traj_pixel[1])), 4, sample_color[jj] , -1)
                        #     #for q in range(len(xyz_list)-1):
                        #         # point_traj_xyz_prev = xyz_list[q][sample_indice].tolist()
                        #         # point_traj_xyz_cur = xyz_list[q+1][sample_indice].tolist()
                        #     #point_traj_xyz_prev = xyz_list[0][sample_indice].tolist()
                        #     point_traj_xyz_cur = xyz_list[len(xyz_list)-1][sample_indice].tolist()
                            
                        #     #assert len(point_traj_xyz_prev)==len(point_traj_xyz_cur)
                        #     for jj in range(len(point_traj_xyz_cur)):
                        #         #point_traj_pixel_prev,point_traj_depth_prev = cam.world_to_pixel(point_traj_xyz_prev[jj])
                        #         point_traj_pixel_cur,point_traj_depth_cur = cam.world_to_pixel(point_traj_xyz_cur[jj])
                        #         #if point_traj_pixel_prev is not None and point_traj_depth_prev < rendered_depth[0,0,int(point_traj_pixel_prev[1]*2),int(point_traj_pixel_prev[0]*2)]+0.1 and \
                        #         if point_traj_pixel_cur is not None and detph_test_flag[jj]:#point_traj_depth_cur < rendered_depth[0,0,int(point_traj_pixel_cur[1]*2),int(point_traj_pixel_cur[0]*2)]+0.1:
                                    
                        #             sample_color = point_color[point_color_indice[jj]]
                        #             #cv2.circle(img, (int(point_traj_pixel_prev[0]*2),int(point_traj_pixel_prev[1]*2)), 8, sample_color , -1 ,cv2.LINE_AA)
                        #             cv2.circle(img, (int(point_traj_pixel_cur[0]*2),int(point_traj_pixel_cur[1]*2)), 8, sample_color , -1,  cv2.LINE_AA)
                        #             # cv2.line(img, (int(point_traj_pixel_prev[0]),int(point_traj_pixel_prev[1])), (int(point_traj_pixel_cur[0]),int(point_traj_pixel_cur[1]))
                        #             #          , color=sample_color[jj], thickness=2, lineType=cv2.LINE_AA)
                        #             #overlay = img.copy()
                        #             #alpha = 0.5
                        #             #cv2.line(overlay, (int(point_traj_pixel_prev[0]*2),int(point_traj_pixel_prev[1]*2)), (int(point_traj_pixel_cur[0]*2),int(point_traj_pixel_cur[1]*2)), sample_color, 6)
                        #             #img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                        #             #cv2.circle(img_res, (int(point_traj_pixel_prev[0]*2),int(point_traj_pixel_prev[1]*2)), 8, sample_color , -1 ,cv2.LINE_AA)
                        #             #cv2.circle(img_res, (int(point_traj_pixel_cur[0]*2),int(point_traj_pixel_cur[1]*2)), 8, sample_color , -1,  cv2.LINE_AA)
                        #             #overlay = img_res.copy()
                        #             #alpha = 0.5
                        #             #cv2.line(overlay, (int(point_traj_pixel_prev[0]*2),int(point_traj_pixel_prev[1]*2)), (int(point_traj_pixel_cur[0]*2),int(point_traj_pixel_cur[1]*2)), sample_color, 6)
                        #             #img_res = cv2.addWeighted(overlay, alpha, img_res, 1 - alpha, 0)

                        #     #cv2.imwrite(f"{self.output_dir}/optim_{n}/cam_{i+1}_{cam.image_name}_point_traj.png",img)
                        #     #cv2.imwrite(f"{self.output_dir}/optim_{n}/cam_{i+1}_{cam.image_name}_point_traj_res.png",img)
                           
                        #     frame_path = os.path.join(self.output_dir,"frames")
                        #     if not os.path.exists(frame_path):
                        #         os.makedirs(frame_path)
                        #     cv2.imwrite(f"{frame_path}/cam_{i+1}_{cam.image_name}_frame{n:05}.png",img)
                        #endregion


                    #create_video_from_images(f"{self.output_dir}/optim_{n}",f"{self.output_dir}/optim_{n}.mp4")

                    #NOTE:save point transform
                    # xyz, features, opacity, scales, rotations = gaussians_optim()
                    # xyz_normalized = gaussians_optim.normalize_xyz(xyz.detach())
                    # if self.use_knn:
                    #     delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
                    # else:
                    #     delta_xyz = self.predict_delta_xyz(xyz_normalized,gaussians_optim.get_3dmask())
                    # # xyz_transform = xyz.detach()+delta_xyz
                    # mask_3d_and_knn_mask = ~(gaussians_optim.get_othermask())
                    # #export_points_for_gaussians(f"{self.output_dir}/optim_{n}.ply", xyz, mask_3d_and_knn_mask, delta_xyz=delta_xyz,  use_triplane=True)

                    if n==self.triplane_optim_iter:
                        gaussians = (xyz + delta_xyz, features, opacity, scales, rotations)
                        gaussians = [p.detach() for p in gaussians]
                        export_ply_for_gaussians(f"{self.output_dir}/result_{n}", gaussians)


        #create_video_from_images(frame_path,f"{self.output_dir}/point.mp4",fps=60)
    
        #ouput gaussians
        with torch.no_grad():
            xyz, features, opacity, scales, rotations = gaussians_optim()
            xyz_normalized = gaussians_optim.normalize_xyz(xyz.detach())
            if self.use_knn:
                delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
            else:
                delta_xyz = self.predict_delta_xyz(xyz_normalized,gaussians_optim.get_3dmask())
            gaussians = (xyz + delta_xyz, features, opacity, scales, rotations)
            #is_visible = gaussians_optim.is_visible.bool()
            gaussians = [p.detach() for p in gaussians]
            # export compare video
            create_video_from_two_folders(f"{self.output_dir}/init", f"{self.output_dir}/optim_{self.total_iterations-1}", f"{self.output_dir}/compare.mp4")
            export_ply_for_gaussians(f"{self.output_dir}/result", gaussians)

    def train_drag_webui(self,webui,update_view_interval = 10):
        gaussians = webui.gaussian
        colmap_cameras = webui.train_colmap_cameras
        camera_extent = webui.cameras_extent
        edit_mask = webui.edit_mask
        handle_points = webui.handle_points
        target_points = webui.target_points

        batch_size = self.batch_size
        gaussians_init = copy.deepcopy(gaussians)

        xyz, features, opacity, scales, rotations = gaussians_init
        gs_init_mask = (xyz[edit_mask],features[edit_mask],opacity[edit_mask],scales[edit_mask],rotations[edit_mask])

        xyz, features, opacity, scales, rotations = gaussians
        gs_mask = opacity[..., 0] >= self.opacity_threshold

        xyz_original = xyz[gs_mask]
        features_original = features[gs_mask]
        opacity_original = opacity[gs_mask]
        scales_original = scales[gs_mask]
        rotations_original = rotations[gs_mask]

        edit_mask = edit_mask[gs_mask]
        gaussians_optim = GaussiansManager(xyz_original, features_original, opacity_original, scales_original, rotations_original, self.lrs, 
                                           edit_mask,self.only_optimize_3d_mask,
                                           use_knn=self.use_knn,knn_coeff=self.knn_coeff,other_coeff=self.other_coeff,knn_number=self.knn_number)

        self.pipe.prepare_embeddings(gaussians,colmap_cameras,
                                     handle_points,target_points,
                                     height=self.image_height,
                                     width=self.image_width,)
        
        with torch.no_grad():
            for i,cam in enumerate(colmap_cameras):  
                xyz, features, opacity, scales, rotations = gaussians_optim()
                gs_optim = (xyz, features, opacity, scales, rotations)
                rendered_image_optim, rendered_depth, rendered_mask = self.renderer([cam],gs_optim, scaling_modifier=1.0, bg_color=None,save_radii_viewspace_points=False,down_sample_res=False)
                import cv2


        for n in tqdm(range(self.total_iterations),desc="trainning drag"):
            xyz, features, opacity, scales, rotations = gaussians_optim()
            xyz_normalized = gaussians_optim.normalize_xyz(xyz.detach())
            
            if self.use_knn:
                delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
            else:
                delta_xyz = self.predict_delta_xyz(xyz_normalized,gaussians_optim.get_3dmask())
            
            #NOTE ablation triplane
            gs_optim = (xyz + delta_xyz, features, opacity, scales, rotations)

            #sample camera
            camera_idx_list = random.sample(range(len(colmap_cameras)), batch_size)
            camera_list = [colmap_cameras[i] for i in camera_idx_list]
     
            #[B,N,2]
            handle_points_pixel_list = []
            target_points_pixel_list = []
            for camera in camera_list:
                #camera = colmap_cameras[i]
                h_pixel = []
                g_pixel = []
                for j in range(len(handle_points)):
                    #swap tensor[0] and tensor[1]
                    #downsample factor 2
                    h_pixel_xy,_ = camera.world_to_pixel(handle_points[j])
                    g_pixel_xy,_ = camera.world_to_pixel(target_points[j])
                    #xy space to hw space
                    h_pixel.append(h_pixel_xy[[1, 0]].unsqueeze(0))
                    g_pixel.append(g_pixel_xy[[1, 0]].unsqueeze(0))
                handle_points_pixel_list.append(torch.cat(h_pixel,dim=0).long())
                target_points_pixel_list.append(torch.cat(g_pixel,dim=0).long())
            
            rendered_image_init, rendered_depth_init, _ = self.renderer(camera_list,gaussians_init)
            rendered_image_optim, rendered_depth_optim, _ = self.renderer(camera_list,gs_optim)
            
            #rendered_depth for mask
            with torch.no_grad():
                _, rendered_depth_mask, _ = self.renderer(camera_list,gs_init_mask,save_radii_viewspace_points=False)
                mask_image = torch.ones_like(rendered_depth_mask)
                mask_image[rendered_depth_mask==camera_list[0].zfar] = 0
                mask_image[rendered_depth_init<rendered_depth_mask-self.mask_depth_treshold] = 0
                mask_image = [mask_image[i].squeeze(0).cpu().numpy().astype(np.uint8) for i in range(mask_image.shape[0])]
                #dilate mask
                #mask_image_dilate = cv2.dilate(mask_image,np.ones((25,25),np.int8),iterations=1)
                import cv2
                drag_mask = [Image.fromarray(cv2.dilate(mask * 255,np.ones((self.mask_dilate_size,self.mask_dilate_size),np.int8),iterations=self.mask_dilate_iter)) for mask in mask_image]
                #drag_mask = np.ones((self.image_height, self.image_width))
                #drag_mask = [Image.fromarray(drag_mask * 255)] * batch_size

            step_ratio = n / self.total_iterations
            #t = torch.full((1,), int(step_ratio ** (1/2) * (self.min_step - self.max_step) + self.max_step), dtype=torch.long, device=self.device)
            #t = torch.full((1,), int(0.5 * (self.max_step - self.min_step)*(1+torch.cos(torch.pi*torch.tensor(step_ratio))) + self.min_step), dtype=torch.long, device=self.device)
            #t = torch.randint(self.min_step, self.max_step, (1,), dtype=torch.long, device=self.device)
            t = torch.full((1,), int(step_ratio * (self.min_step - self.max_step) + self.max_step), dtype=torch.long, device=self.device)
            
            cur_guidance_scale_points = (self.guidance_scale-1.0) * (1.0 - step_ratio) ** 2 + 1.0
            #cur_guidance_scale_points = self.guidance_scale


            loss_latent_sds, loss_image_sds, loss_embedding_lora = \
                self.pipe(
                    ref_image=rendered_image_init,
                    render_image=rendered_image_optim,
                    mask_image=drag_mask,
                    handle_points_pixel_list=handle_points_pixel_list,
                    target_points_pixel_list=target_points_pixel_list,
                    time_step=t,
                    guidance_scale_points=cur_guidance_scale_points,
                    prompt=[""] * batch_size,
                    height=self.image_height,
                    width=self.image_width,
                    num_train_timesteps=self.num_train_timesteps,
                )
            loss_reg = F.l1_loss(delta_xyz[gaussians_optim.get_knnmask()], torch.zeros_like(delta_xyz[gaussians_optim.get_knnmask()]), reduction="mean")            
            loss = loss_latent_sds * self.lambda_latent_sds + loss_image_sds * self.lambda_image_sds + loss_embedding_lora + loss_reg * self.lambda_reg

            loss.backward()
            
            #NOTE Ablation
            if n > self.triplane_optim_iter:
                gaussians_optim.optimizer.step()
            gaussians_optim.optimizer.zero_grad()

            self.grid_net_optimizer.step()
            self.grid_net_optimizer.zero_grad()
            #NOTE Ablation
            if n == self.triplane_optim_iter:
                for param_group in self.grid_net_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.triplane_lr_down_scale
            
            self.pipe.unet_emb_lora_optimizer.step()
            self.pipe.unet_emb_lora_optimizer.zero_grad()
            #lr_scheduler_triplane.step()

            if n > self.triplane_optim_iter and step_ratio > 0.25 and step_ratio <= 0.8:
                for radii, viewspace_points in zip(self.renderer.radii, self.renderer.viewspace_points):
                    visibility_filter = radii > 0
                    gaussians_optim.is_visible[visibility_filter] = 1
                    gaussians_optim.max_radii2D[visibility_filter] = torch.max(gaussians_optim.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians_optim.add_densification_stats(viewspace_points, visibility_filter)
                if n % self.densification_interval == 0 and n != 0:
                    gaussians_optim.densify_and_prune(max_grad=10,extent=camera_extent,opacity_threshold=0.01)
            
            
            with torch.no_grad():
                if (n % 500==0 and n!=0) or n == self.total_iterations - 1 or n==self.triplane_optim_iter:
                    xyz, features, opacity, scales, rotations = gaussians_optim()
                    xyz_normalized = gaussians_optim.normalize_xyz(xyz.detach())
                    # delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
                    if self.use_knn:
                        delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
                    else:
                        delta_xyz = self.predict_delta_xyz(xyz_normalized,gaussians_optim.get_3dmask())
                    gs_optim = (xyz + delta_xyz, features, opacity, scales, rotations)

                    for i,cam in enumerate(colmap_cameras):
                        rendered_image_optim, rendered_depth, rendered_mask = self.renderer([cam],gs_optim, scaling_modifier=1.0, bg_color=None,save_radii_viewspace_points=False)
                        import cv2
                        img = (rendered_image_optim/2 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                        img = (img.clip(min = 0, max = 1)*255.0).astype(np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        if not os.path.exists(f"{self.output_dir}/optim_{n}/"):
                            os.makedirs(f"{self.output_dir}/optim_{n}/")
                        cv2.imwrite(f"{self.output_dir}/optim_{n}/cam_{i+1}_{cam.image_name}.png",img)

                    if n==self.triplane_optim_iter:
                        gaussians = (xyz + delta_xyz, features, opacity, scales, rotations)
                        gaussians = [p.detach() for p in gaussians]
                        export_ply_for_gaussians(f"{self.output_dir}/result_{n}", gaussians)
                #update viewer
                if (n%update_view_interval==0 and n!=0):
                    xyz, features, opacity, scales, rotations = gaussians_optim()
                    xyz_normalized = gaussians_optim.normalize_xyz(xyz.detach())
                    # delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
                    if self.use_knn:
                        delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
                    else:
                        delta_xyz = self.predict_delta_xyz(xyz_normalized,gaussians_optim.get_3dmask())
                    webui.gaussian = (xyz + delta_xyz, features, opacity, scales, rotations)
                    webui.edit_mask = gaussians_optim.get_3dmask()

        #ouput gaussians
        with torch.no_grad():
            xyz, features, opacity, scales, rotations = gaussians_optim()
            xyz_normalized = gaussians_optim.normalize_xyz(xyz.detach())
            if self.use_knn:
                delta_xyz = self.predict_delta_xyz(xyz_normalized,~gaussians_optim.get_othermask())
            else:
                delta_xyz = self.predict_delta_xyz(xyz_normalized,gaussians_optim.get_3dmask())
            gaussians = (xyz + delta_xyz, features, opacity, scales, rotations)
            #is_visible = gaussians_optim.is_visible.bool()
            gaussians = [p.detach() for p in gaussians]
            # export compare video
            create_video_from_two_folders(f"{self.output_dir}/init", f"{self.output_dir}/optim_{self.total_iterations-1}", f"{self.output_dir}/compare.mp4")
            export_ply_for_gaussians(f"{self.output_dir}/result", gaussians)
    
    def train_drag_2d(self, gaussians, colmap_cameras, edit_mask, handle_points, target_points):
        #batch_size = self.batch_size
        batch_size = 8
        gaussians_init = copy.deepcopy(gaussians)

        xyz, features, opacity, scales, rotations = gaussians_init
        gs_init_mask = (xyz[edit_mask],features[edit_mask],opacity[edit_mask],scales[edit_mask],rotations[edit_mask])

        xyz, features, opacity, scales, rotations = gaussians
        gs_mask = opacity[..., 0] >= self.opacity_threshold

        xyz_original = xyz[gs_mask]
        features_original = features[gs_mask]
        opacity_original = opacity[gs_mask]
        scales_original = scales[gs_mask]
        rotations_original = rotations[gs_mask]

        edit_mask = edit_mask[gs_mask]
        #gaussians_optim = GaussiansManager(xyz_original, features_original, opacity_original, scales_original, rotations_original, self.lrs, edit_mask)

        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gaussians_optim.optimizer, gamma=(self.lr_scale_end / self.lr_scale) ** (1 / self.total_iterations))
        gs_optim = (xyz_original, features_original, opacity_original, scales_original, rotations_original)

        self.pipe.prepare_embeddings(gaussians,colmap_cameras,
                             handle_points,target_points,
                             height=self.image_height,
                             width=self.image_width,)
        for k in range(0,len(colmap_cameras),batch_size * 10):
            #sample camera
            camera_idx_list = [idx for idx in range(k,k + batch_size * 10,10)]
            camera_list = [colmap_cameras[i] for i in camera_idx_list]

            #[B,N,2]
            handle_points_pixel_list = []
            target_points_pixel_list = []
            for camera in camera_list:
                #camera = colmap_cameras[i]
                h_pixel = []
                g_pixel = []
                for j in range(len(handle_points)):
                    #swap tensor[0] and tensor[1]
                    #downsample factor 2
                    h_pixel_xy,_ = camera.world_to_pixel(handle_points[j])
                    g_pixel_xy,_ = camera.world_to_pixel(target_points[j])
                    #xy space to hw space
                    h_pixel.append(h_pixel_xy[[1, 0]].unsqueeze(0))
                    g_pixel.append(g_pixel_xy[[1, 0]].unsqueeze(0))
                handle_points_pixel_list.append(torch.cat(h_pixel,dim=0).long())
                target_points_pixel_list.append(torch.cat(g_pixel,dim=0).long())

            rendered_image_init, rendered_depth_init, _ = self.renderer(camera_list,gaussians_init)
            rendered_image_optim, rendered_depth_optim, _ = self.renderer(camera_list,gs_optim)

            latents = self.pipe.vae.encode(rendered_image_optim.to(dtype=self.pipe.vae.dtype)).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
 
            latents = latents.clone().float().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([latents], lr=0.005)

            total_iteration = 2000
            for n in tqdm(range(total_iteration),desc="trainning drag 2d"):
                #gs_optim_mask = xyz[0][gaussians_optim.edit_mask],features[0][gaussians_optim.edit_mask],opacity[0][gaussians_optim.edit_mask],scales[0][gaussians_optim.edit_mask],rotations[0][gaussians_optim.edit_mask]
                #cam_idx = torch.randint(0, len(colmap_cameras),(1,))
                #camera = colmap_cameras[cam_idx]
                #rendered_depth for mask
                with torch.no_grad():
                    _, rendered_depth_mask, _ = self.renderer(camera_list,gs_init_mask,save_radii_viewspace_points=False)
                    mask_image = torch.ones_like(rendered_depth_mask)
                    mask_image[rendered_depth_mask==camera_list[0].zfar] = 0
                    mask_image[rendered_depth_init<rendered_depth_mask-self.mask_depth_treshold] = 0
                    mask_image = [mask_image[i].squeeze(0).cpu().numpy().astype(np.uint8) for i in range(mask_image.shape[0])]
                    #dilate mask
                    #mask_image_dilate = cv2.dilate(mask_image,np.ones((25,25),np.int8),iterations=1)
                    import cv2
                    drag_mask = [Image.fromarray(cv2.dilate(mask * 255,np.ones((self.mask_dilate_size,self.mask_dilate_size),np.int8),iterations=self.mask_dilate_iter)) for mask in mask_image]
                    #drag_mask = np.ones((self.image_height, self.image_width))
                    #drag_mask = [Image.fromarray(drag_mask * 255)] * batch_size

                step_ratio = n / self.total_iterations

                #t = torch.full((1,), int(step_ratio ** (1/2) * (self.min_step - self.max_step) + self.max_step), dtype=torch.long, device=self.device)
                t = torch.full((1,), int(0.5 * (self.max_step - self.min_step)*(1+torch.cos(torch.pi*torch.tensor(step_ratio))) + self.min_step), dtype=torch.long, device=self.device)
                #t = torch.randint(self.min_step, self.max_step, (1,), dtype=torch.long, device=self.device)
                cur_guidance_scale_points = (self.guidance_scale-1.0) * (1.0 - step_ratio) ** 2 + 1.0
                #cur_guidance_scale_points = self.guidance_scale

                loss_latent_sds, loss_image_sds, loss_embedding_lora = \
                    self.pipe.compute_sdspp_loss_2d(
                        ref_image=rendered_image_init,
                        render_latent=latents.half(),
                        mask_image=drag_mask,
                        camera_idx=camera_idx_list,
                        time_step=t,
                        prompt=[""] * batch_size,
                        height=self.image_height,
                        width=self.image_width,
                        num_train_timesteps=self.num_train_timesteps,
                        guidance_scale_points=cur_guidance_scale_points,
                        num_guidance_steps=None,
                        num_images_per_prompt=1, # set this to 4 as we are generating 4 candidate results
                        output_type='pt',
                        handle_points_pixel_list=handle_points_pixel_list,
                        target_points_pixel_list=target_points_pixel_list,
                        skip_cfg_appearance_encoder=False
                    )
                loss = loss_latent_sds * self.lambda_latent_sds + loss_image_sds * self.lambda_image_sds + loss_embedding_lora


                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                self.pipe.unet_emb_lora_optimizer.step()
                self.pipe.unet_emb_lora_optimizer.zero_grad()


                if n % 500 == 0 or n == total_iteration - 1:
                    with torch.no_grad():
                        for i in range(len(camera_idx_list)):
                            cam_idx = camera_idx_list[i]
                            cam = camera_list[i]
                            rendered_image_optim = self.pipe.vae.decode(latents[i:i+1,:,:,:].to(self.pipe.vae.dtype) / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
                            import cv2
                            img = (rendered_image_optim/2 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                            img = (img.clip(min = 0, max = 1)*255.0).astype(np.uint8)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            assert len(handle_points) == len(target_points)
                            for j in range(len(handle_points)):
                                h_pixel,_ = cam.world_to_pixel(handle_points[j])
                                g_pixel,_ = cam.world_to_pixel(target_points[j])
                                color = (0,0,255) if j == 0 else ((0,255,0) if j == 1 else (255,0,0))
                                cv2.arrowedLine(img, (int(h_pixel[0]),int(h_pixel[1])), 
                                                    (int(g_pixel[0]),int(g_pixel[1])), color)
                            if not os.path.exists(f"{self.output_dir}/cam_{cam_idx}/"):
                                os.makedirs(f"{self.output_dir}/cam_{cam_idx}/")
                            cv2.imwrite(f"{self.output_dir}/cam_{cam_idx}/optim_{n}.png",img)

    def train_drag_2d_denoise(self, gaussians, colmap_cameras, edit_mask, handle_points, target_points):
        #batch_size = self.batch_size
        batch_size = 1
        gaussians_init = copy.deepcopy(gaussians)

        xyz, features, opacity, scales, rotations = gaussians_init
        gs_init_mask = (xyz[edit_mask],features[edit_mask],opacity[edit_mask],scales[edit_mask],rotations[edit_mask])

        xyz, features, opacity, scales, rotations = gaussians
        gs_mask = opacity[..., 0] >= self.opacity_threshold

        xyz_original = xyz[gs_mask]
        features_original = features[gs_mask]
        opacity_original = opacity[gs_mask]
        scales_original = scales[gs_mask]
        rotations_original = rotations[gs_mask]

        edit_mask = edit_mask[gs_mask]
        #gaussians_optim = GaussiansManager(xyz_original, features_original, opacity_original, scales_original, rotations_original, self.lrs, edit_mask)

        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gaussians_optim.optimizer, gamma=(self.lr_scale_end / self.lr_scale) ** (1 / self.total_iterations))
        gs_optim = (xyz_original, features_original, opacity_original, scales_original, rotations_original)

        self.pipe.prepare_embeddings(gaussians,colmap_cameras,
                             handle_points,target_points,
                             height=self.image_height,
                             width=self.image_width,)
        
        sample_interval = 1
        for k in range(0,len(colmap_cameras),batch_size * sample_interval):
            #sample camera
            camera_idx_list = [idx for idx in range(k,k + batch_size * sample_interval,sample_interval)]
            camera_list = [colmap_cameras[i] for i in camera_idx_list]

            #[B,N,2]
            handle_points_pixel_list = []
            target_points_pixel_list = []
            for camera in camera_list:
                #camera = colmap_cameras[i]
                h_pixel = []
                g_pixel = []
                for j in range(len(handle_points)):
                    #swap tensor[0] and tensor[1]
                    #downsample factor 2
                    h_pixel_xy,_ = camera.world_to_pixel(handle_points[j])
                    g_pixel_xy,_ = camera.world_to_pixel(target_points[j])
                    #xy space to hw space
                    h_pixel.append(h_pixel_xy[[1, 0]].unsqueeze(0))
                    g_pixel.append(g_pixel_xy[[1, 0]].unsqueeze(0))
                handle_points_pixel_list.append(torch.cat(h_pixel,dim=0).long())
                target_points_pixel_list.append(torch.cat(g_pixel,dim=0).long())

            rendered_image_init, rendered_depth_init, _ = self.renderer(camera_list,gaussians_init)
            rendered_image_optim, rendered_depth_optim, _ = self.renderer(camera_list,gs_optim)

            latents = self.pipe.vae.encode(rendered_image_optim.to(dtype=self.pipe.vae.dtype)).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor

            with torch.no_grad():
                _, rendered_depth_mask, _ = self.renderer(camera_list,gs_init_mask,save_radii_viewspace_points=False)
                mask_image = torch.ones_like(rendered_depth_mask)
                mask_image[rendered_depth_mask==camera_list[0].zfar] = 0
                mask_image[rendered_depth_init<rendered_depth_mask-self.mask_depth_treshold] = 0
                mask_image = [mask_image[i].squeeze(0).cpu().numpy().astype(np.uint8) for i in range(mask_image.shape[0])]
                #dilate mask
                #mask_image_dilate = cv2.dilate(mask_image,np.ones((25,25),np.int8),iterations=1)
                import cv2
                drag_mask = [Image.fromarray(cv2.dilate(mask * 255,np.ones((self.mask_dilate_size,self.mask_dilate_size),np.int8),iterations=self.mask_dilate_iter)) for mask in mask_image]
                #drag_mask = np.ones((self.image_height, self.image_width))
                #drag_mask = [Image.fromarray(drag_mask * 255)] * batch_size
            
            #cur_guidance_scale_points = (self.guidance_scale-1.0) * (1.0 - step_ratio) ** 2 + 1.0
            #cur_guidance_scale_points = self.guidance_scale

            denoise_num_inference_steps = 25 if self.lcm_lora_path is None else 8
            denoise_guidance_scale_points = 4.0 if self.lcm_lora_path is None else 3.0

            denoised_latent,latents_noisy_750,latents_750_pred_x0 = self.pipe.denoise_2d(
                                ref_image=rendered_image_init,
                                render_latent=latents,
                                mask_image=drag_mask,
                                camera_idx=camera_idx_list,
                                prompt=[""] * batch_size,
                                height=self.image_height,
                                width=self.image_width,
                                num_train_timesteps=self.num_train_timesteps,

                                guidance_scale_points=denoise_guidance_scale_points,
                                num_inference_steps = denoise_num_inference_steps,

                                num_guidance_steps=None,
                                num_images_per_prompt=1, # set this to 4 as we are generating 4 candidate results
                                output_type='pt',
                                handle_points_pixel_list=handle_points_pixel_list,
                                target_points_pixel_list=target_points_pixel_list,
                                skip_cfg_appearance_encoder=False,
                                )
            

            with torch.no_grad():
                img_list = []
                img_750_list = []
                img_750_pred_x0_list = []
                for i in range(len(camera_idx_list)):
                    cam_idx = camera_idx_list[i]
                    cam = camera_list[i]
                    rendered_image_optim = self.pipe.vae.decode(denoised_latent[i:i+1,:,:,:].to(self.pipe.vae.dtype) / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
                    image_750 = self.pipe.vae.decode(latents_noisy_750[i:i+1,:,:,:].to(self.pipe.vae.dtype) / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
                    image_750_pred_x0 = self.pipe.vae.decode(latents_750_pred_x0[i:i+1,:,:,:].to(self.pipe.vae.dtype) / self.pipe.vae.config.scaling_factor, return_dict=False)[0]

                    import cv2
                    img = (rendered_image_optim/2 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    img = (img.clip(min = 0, max = 1)*255.0).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    assert len(handle_points) == len(target_points)
                    #draw line
                    # for j in range(len(handle_points)):
                    #     h_pixel,_ = cam.world_to_pixel(handle_points[j])
                    #     g_pixel,_ = cam.world_to_pixel(target_points[j])
                    #     color = (0,0,255) if j == 0 else ((0,255,0) if j == 1 else (255,0,0))
                    #     cv2.arrowedLine(img, (int(h_pixel[0]),int(h_pixel[1])), 
                    #                         (int(g_pixel[0]),int(g_pixel[1])), color)
                    img_list.append(img)

                    image_750 = (image_750/2 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    image_750 = (image_750.clip(min = 0, max = 1)*255.0).astype(np.uint8)
                    image_750 = cv2.cvtColor(image_750, cv2.COLOR_RGB2BGR)
                    img_750_list.append(image_750)

                    image_750_pred_x0 = (image_750_pred_x0/2 + 0.5).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    image_750_pred_x0 = (image_750_pred_x0.clip(min = 0, max = 1)*255.0).astype(np.uint8)
                    image_750_pred_x0 = cv2.cvtColor(image_750_pred_x0, cv2.COLOR_RGB2BGR)
                    img_750_pred_x0_list.append(image_750_pred_x0)

                #img (h,w,c)
                img = np.concatenate(img_list, axis=1)
                image_750 = np.concatenate(img_750_list, axis=1)
                img_750_pred_x0 = np.concatenate(img_750_pred_x0_list, axis=1)
                if not os.path.exists(f"{self.output_dir}/"):
                    os.makedirs(f"{self.output_dir}/")
                cv2.imwrite(f"{self.output_dir}/cam_{camera_idx_list[-1]+1}.png",img)
                #cv2.imwrite(f"{self.output_dir}/cam_{camera_idx_list[-1]+1}_750.png",image_750)
                #cv2.imwrite(f"{self.output_dir}/cam_{camera_idx_list[-1]+1}_750_pred_x0.png",img_750_pred_x0)


