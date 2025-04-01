# *************************************************************************
# Copyright (2024) Bytedance Inc.
#
# Copyright (2024) LightningDrag Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import PIL.Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor

from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.models.attention import BasicTransformerBlock

from models.appearance_encoder import AppearanceEncoderModel
from models.mutual_self_attention import ReferenceAttentionControl
from models.attention_processor import PointEmbeddingAttnProcessor, IPAttnProcessor
from einops import rearrange

from drag_util import torch_dfs

def points_to_disk_map(points, H, W):
    """
    Convert a set of points into a two-dimensional disk map with shape (H, W).
    Args:
        points (numpy.ndarray): Array of shape (N, 2) representing (H, W) coordinates.
        H (int): Height of the disk map.
        W (int): Width of the disk map.
    Returns:
        numpy.ndarray: Two-dimensional disk map with shape (H, W).
    """
    # Create an empty disk map
    disk_map = torch.zeros((H, W)).long().to(points.device)

    if len(points) == 0:
        return disk_map

    # Assign values to disk map
    idx = torch.arange(len(points)).to(points.device) + 1
    disk_map[points[:, 0].long(), points[:, 1].long()] = idx

    return disk_map


# TODO: replace with prepare_ref_latents()
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class DragPipeline(StableDiffusionPipeline):
    def __init__(self, 
                 vae: AutoencoderKL, 
                 text_encoder: CLIPTextModel, 
                 tokenizer: CLIPTokenizer, 
                 unet: UNet2DConditionModel, 
                 appearance_encoder: AppearanceEncoderModel,
                 scheduler: KarrasDiffusionSchedulers, 
                 safety_checker: StableDiffusionSafetyChecker, 
                 feature_extractor: CLIPImageProcessor = None,
                 image_encoder: CLIPVisionModelWithProjection = None,
                 point_embedding = None,
                 image_proj_model = None,
                 fusion_blocks = "midup",
                 requires_safety_checker: bool = True,
                 use_norm_attn_processor: bool = False,
                 initialize_attn_processor: bool = False, # whether to reinit attn processor
                 num_ip_tokens = 4,
                 initialize_ip_attn_processor: bool = False,
        ):
        super().__init__(vae,
                         text_encoder,
                         tokenizer,
                         unet,
                         scheduler,
                         safety_checker,
                         feature_extractor,
                         image_encoder,
                         requires_safety_checker)
        self.appearance_encoder = appearance_encoder
        self.point_embedding = point_embedding
        self.fusion_blocks = fusion_blocks
        self.num_ip_tokens = num_ip_tokens

        # Setup attention processor
        self.use_norm_attn_processor = use_norm_attn_processor
        if initialize_attn_processor:
            self.set_up_point_attn_processor()

        if initialize_ip_attn_processor:
            self.set_up_ip_attn_processor()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            appearance_encoder=appearance_encoder,
            point_embedding=point_embedding,
            image_proj_model=image_proj_model,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device).to(self.unet.dtype)
        
    def set_up_point_attn_processor(self):
        device, dtype = self.unet.conv_in.weight.device, self.unet.conv_in.weight.dtype
        scale_idx = 0
        # downsample ratio of point embeddings: [8x, 16x, 32x, 64x]
        for down_block in self.unet.down_blocks:
            if self.fusion_blocks == "full":
                for m in torch_dfs(down_block):
                    if isinstance(m, BasicTransformerBlock):
                        if type(self.point_embedding.output_dim) == int:
                            embed_dim = self.point_embedding.output_dim
                        else:
                            embed_dim = self.point_embedding.output_dim[scale_idx]
                        processor = PointEmbeddingAttnProcessor(
                            embed_dim=embed_dim,
                            hidden_size=m.attn1.to_q.out_features,
                            use_norm=self.use_norm_attn_processor).to(device, dtype)
                        processor.requires_grad_(False)
                        processor.eval()
                        m.attn1.processor = processor
            if down_block.downsamplers is not None:
                scale_idx += 1

        if self.fusion_blocks == "full" or self.fusion_blocks == "midup":
            for m in torch_dfs(self.unet.mid_block):
                if isinstance(m, BasicTransformerBlock):
                    if type(self.point_embedding.output_dim) == int:
                        embed_dim = self.point_embedding.output_dim
                    else:
                        embed_dim = self.point_embedding.output_dim[scale_idx]
                    processor = PointEmbeddingAttnProcessor(
                        embed_dim=embed_dim,
                        hidden_size=m.attn1.to_q.out_features,
                        use_norm=self.use_norm_attn_processor).to(device, dtype)
                    processor.requires_grad_(False)
                    processor.eval()
                    m.attn1.processor = processor

        for up_block in self.unet.up_blocks:
            for m in torch_dfs(up_block):
                if isinstance(m, BasicTransformerBlock):
                    if type(self.point_embedding.output_dim) == int:
                        embed_dim = self.point_embedding.output_dim
                    else:
                        embed_dim = self.point_embedding.output_dim[scale_idx]
                    processor = PointEmbeddingAttnProcessor(
                        embed_dim=embed_dim,
                        hidden_size=m.attn1.to_q.out_features,
                        use_norm=self.use_norm_attn_processor).to(device, dtype)
                    processor.requires_grad_(False)
                    processor.eval()
                    m.attn1.processor = processor
            if up_block.upsamplers is not None:
                scale_idx -= 1

    def set_up_ip_attn_processor(self):
        for m in torch_dfs(self.unet):
            if isinstance(m, BasicTransformerBlock):
                processor = IPAttnProcessor(
                    hidden_size=m.attn2.to_q.in_features,
                    cross_attention_dim=self.unet.config.cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_ip_tokens,
                )
                processor.requires_grad_(False)
                processor.eval()
                m.attn2.processor = processor

    def prepare_point_embeddings(
        self,
        batch_size,
        device,
        dtype,
        point_embedding,
        handle_points, 
        target_points, 
        height, 
        width, 
        do_classifier_free_guidance,
    ):
        handle_disk_map = points_to_disk_map(handle_points, height, width)
        target_disk_map = points_to_disk_map(target_points, height, width)

        handle_disk_map = handle_disk_map.to(device, dtype) # (H, W)
        target_disk_map = target_disk_map.to(device, dtype) # (H, W)

        handle_disk_map = handle_disk_map.unsqueeze(dim=0) # (1, H, W)
        target_disk_map = target_disk_map.unsqueeze(dim=0) # (1, H, W)

        if do_classifier_free_guidance:
            # repeat in batch dimension if we need to do CFG
            handle_disk_map = torch.repeat_interleave(handle_disk_map, 2, dim=0)
            target_disk_map = torch.repeat_interleave(target_disk_map, 2, dim=0)

        handle_disk_map = handle_disk_map.unsqueeze(dim=1)
        target_disk_map = target_disk_map.unsqueeze(dim=1)
        handle_embeddings, target_embeddings = \
            point_embedding(handle_disk_map, target_disk_map)

        # repeat if needed
        if handle_embeddings[0].shape[0] < batch_size:
            assert batch_size % handle_embeddings[0].shape[0] == 0, \
                "shape mismatch with batch size"
            num_img = handle_embeddings[0].shape[0]
            handle_embeddings = [
                h.repeat(batch_size//num_img, 1, 1, 1)
                for h in handle_embeddings
            ]
            target_embeddings = [
                t.repeat(batch_size//num_img, 1, 1, 1)
                for t in target_embeddings
            ]

        return handle_embeddings, target_embeddings

    # TODO: replace with prepare_ref_latents()
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    # from https://github.com/huggingface/diffusers/blob/9941f1f61b5d069ebedab92f3f620a79cc043ef2/src/diffusers/pipelines/controlnet/pipeline_controlnet.py#L794
    def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
        ):
            # image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
            image = (image + 1) / 2 
            image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

            image_batch_size = image.shape[0]

            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                # image batch size is the same as prompt batch size
                repeat_by = num_images_per_prompt

            image = image.repeat_interleave(repeat_by, dim=0)

            image = image.to(device=device, dtype=dtype)

            if do_classifier_free_guidance and not guess_mode:
                image = torch.cat([image] * 2)

            return image

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def prepare_ref_latents(
        self,
        refimage,
        batch_size,
        dtype,
        device,
        generator,
        do_classifier_free_guidance
    ):
        refimage = refimage.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            ref_image_latents = [
                self.vae.encode(refimage[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            ref_image_latents = torch.cat(ref_image_latents, dim=0)
        else:
            ref_image_latents = self.vae.encode(refimage).latent_dist.sample(generator=generator)
        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents

        # duplicate mask and ref_image_latents for each generation per prompt, using mps friendly method
        if ref_image_latents.shape[0] < batch_size:
            if not batch_size % ref_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {ref_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            ref_image_latents = ref_image_latents.repeat(batch_size // ref_image_latents.shape[0], 1, 1, 1)

        ref_image_latents = torch.cat([ref_image_latents] * 2) if do_classifier_free_guidance else ref_image_latents

        # aligning device to prevent device errors when concating it with the latent model input
        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
        return ref_image_latents

    def prepare_point_embeddings_per_batch(self,handle_points_pixel_list,target_points_pixel_list,height,width):
        handle_embeddings_list_tmp,target_embeddings_list_tmp =[],[]

        for handle_points_pixel,target_points_pixel in zip(handle_points_pixel_list,target_points_pixel_list):
            handle_embeddings, target_embeddings = self.prepare_point_embeddings(
                #batch_size * num_images_per_prompt,
                1,
                'cuda',
                torch.float16,
                self.point_embedding,
                handle_points_pixel, 
                target_points_pixel, 
                height, 
                width, 
                do_classifier_free_guidance=False, 
            )
            handle_embeddings_list_tmp.append(handle_embeddings)
            target_embeddings_list_tmp.append(target_embeddings)

        handle_embeddings_list,target_embeddings_list = [],[]

        for j in range(len(handle_embeddings_list_tmp[0])):
            handle_embeddings_list.append(torch.cat([emb[j] for emb in handle_embeddings_list_tmp],dim=0))
            target_embeddings_list.append(torch.cat([emb[j] for emb in target_embeddings_list_tmp],dim=0))
        
        return handle_embeddings_list,target_embeddings_list


    def prepare_embeddings(self,gaussians,colmap_cameras,
                           handle_points,target_points,
                           height,width):
        # 3. set up the reference attention control mechanism
        # batch_size = 1
        # num_images_per_prompt = 1
        # skip_cfg_appearance_encoder = False
        # do_classifier_free_guidance = True
        
        if self.fusion_blocks == "midup":
            self.attn_modules = [module for module in
                            (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks))
                            if isinstance(module, BasicTransformerBlock)]
        elif self.fusion_blocks == "up":
            self.attn_modules = [module for module in
                            torch_dfs(self.unet.up_blocks)
                            if isinstance(module, BasicTransformerBlock)]
        elif self.fusion_blocks == "full":
            self.attn_modules = [module for module in
                            torch_dfs(self.unet)
                            if isinstance(module, BasicTransformerBlock)]            
        else:
            raise NotImplementedError(f"fusion blocks {self.fusion_blocks} not implemented")
        self.attn_modules = sorted(self.attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
        
        #region origin point emb
        # self.handle_embeddings_list = []
        # self.target_embeddings_list = []

        # for i in range(len(colmap_cameras)):
        #     camera = colmap_cameras[i]

        #     h_pixel = []
        #     g_pixel = []
        #     for j in range(len(handle_points)):
        #         #swap tensor[0] and tensor[1]
        #         #downsample factor 2
        #         h_pixel_xy,_ = camera.world_to_pixel(handle_points[j])
        #         g_pixel_xy,_ = camera.world_to_pixel(target_points[j])
        #         #xy space to hw space
        #         h_pixel.append(h_pixel_xy[[1, 0]].unsqueeze(0))
        #         g_pixel.append(g_pixel_xy[[1, 0]].unsqueeze(0))
        #     handle_points_pixel = torch.cat(h_pixel,dim=0).long()
        #     target_points_pixel = torch.cat(g_pixel,dim=0).long()

        #     handle_embeddings, target_embeddings = self.prepare_point_embeddings(
        #         batch_size * num_images_per_prompt,
        #         'cuda',
        #         torch.float16,
        #         self.point_embedding,
        #         handle_points_pixel, 
        #         target_points_pixel, 
        #         height, 
        #         width, 
        #         do_classifier_free_guidance=False, 
        #     )
        #     self.handle_embeddings_list.append(handle_embeddings)
        #     self.target_embeddings_list.append(target_embeddings)
        #endregion

        # pipe = StableDiffusionPipeline.from_pretrained(
        #     'stabilityai/stable-diffusion-2-1-base', local_files_only=True
        # )
        pipe = StableDiffusionPipeline.from_pretrained(
             'botp/stable-diffusion-v1-5', local_files_only=False
        )

        self.learnable_embeddings = nn.Parameter(torch.zeros([1,77,768]).detach().cuda().float().clone())
        import copy
        #tgt inpainting
        #self.unet_lora = copy.deepcopy(self.unet)
        self.unet_lora = pipe.unet.requires_grad_(False).to(self.device).to(self.unet.dtype)
        del pipe
        
        from peft import LoraConfig, get_peft_model
        lora_rank = 16
        lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
        self.unet_lora = get_peft_model(self.unet_lora, lora_config)
        self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet_lora.parameters()))
        from diffusers.training_utils import cast_training_params
        cast_training_params(self.unet_lora, dtype=torch.float32)

        self.unet_emb_lora_optimizer = torch.optim.AdamW([
                {"name": "embeddings", 'params':  self.learnable_embeddings, 'lr': 1e-3},
                {"name": "lora_layers", 'params': self.lora_layers, 'lr': 5e-4},
            ])
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        ref_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        render_image: torch.FloatTensor = None,
        mask_image: PIL.Image.Image = None,
        handle_points_pixel_list = None,
        target_points_pixel_list = None,
        time_step = None,
        guidance_scale_points: float = 3.0,
        height: int = 512,
        width: int = 512,
        num_train_timesteps: int = 1000,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        skip_cfg_appearance_encoder: bool = False
    ):

        assert self.fusion_blocks in ["midup", "full", "up"]
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale_points > 1.0
        with torch.no_grad():
            reference_control_writer = ReferenceAttentionControl(
                                            self.appearance_encoder,
                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                            skip_cfg_appearance_encoder=skip_cfg_appearance_encoder,
                                            num_images_per_prompt=num_images_per_prompt,
                                            mode='write',
                                            fusion_blocks=self.fusion_blocks,
                                            batch_size = batch_size)
            reference_control_reader = ReferenceAttentionControl(
                                            self.unet,
                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                            skip_cfg_appearance_encoder=skip_cfg_appearance_encoder,
                                            num_images_per_prompt=num_images_per_prompt,
                                            mode='read',
                                            fusion_blocks=self.fusion_blocks,
                                            batch_size = batch_size)
            # 4. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            prompt_embeds_tuple = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )
            prompt_embeds = prompt_embeds_tuple[0]
            negative_prompt_embeds = prompt_embeds_tuple[1]
            # 4.5 Encode input image
            if self.image_encoder is not None:
                # FIXME: make this generalizable!
                #assert ref_image.shape[0] == 1
                #pil_image = rearrange(ref_image[0], 'c h w -> h w c')
                clip_image_embeds_list = []
                import random
                #for j in random.sample(list(range(batch_size)),batch_size):
                for j in range(batch_size):
                    pil_image = rearrange(ref_image[j], 'c h w -> h w c')
                    pil_image = 127.5 * (pil_image + 1)
                    pil_image = [PIL.Image.fromarray(np.uint8(pil_image.detach().cpu()))]
                    clip_image = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values[0].unsqueeze(0)
                    clip_image_embeds = self.image_encoder(clip_image.to(prompt_embeds.dtype).to(self.image_encoder.device)).image_embeds.unsqueeze(1)
                    clip_image_embeds_list.append(clip_image_embeds)
                clip_image_embeds = torch.cat(clip_image_embeds_list,dim=0)
                # image proj model from IP-Adapter
                image_prompt_embeds = self.image_proj_model(clip_image_embeds)
                negative_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
                # repeat to satisfy the batch size
                image_prompt_embeds = image_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
                negative_image_prompt_embeds = negative_image_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
                #NOTE for LoRA
                prompt_embeds = torch.cat([prompt_embeds_tuple[0], image_prompt_embeds], dim=1)
                negative_prompt_embeds = torch.cat([prompt_embeds_tuple[1], negative_image_prompt_embeds], dim=1)
         

            mask_condition = self.mask_processor.preprocess(
                mask_image, height=height, width=width, resize_mode="default", crops_coords=None
            )
            masked_image = ref_image * (mask_condition.cuda() < 0.5)
            mask, masked_image_latents = self.prepare_mask_latents(
                mask_condition,
                masked_image,
                batch_size * num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance=False,
            )

            #tge inpainiting
            # from PIL import Image
            # mask_all_one = [Image.fromarray(np.zeros((312, 512))*225) for mask in mask_image]     
            # mask_condition_all_one = self.mask_processor.preprocess(
            #     mask_all_one, height=height, width=width, resize_mode="default", crops_coords=None
            # )
            # masked_image_all_one = ref_image * (mask_condition_all_one.cuda() < 0.5)            
            # mask_all_one, masked_image_latents_all_one = self.prepare_mask_latents(
            #     mask_condition_all_one,
            #     masked_image_all_one,
            #     batch_size * num_images_per_prompt,
            #     height,
            #     width,
            #     prompt_embeds.dtype,
            #     device,
            #     generator,
            #     do_classifier_free_guidance=False,
            # )


            # 5. Preprocess reference image
            #NOTE: [-1,1] for image preprocess
            ref_image = self.prepare_image(
                image=ref_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=prompt_embeds.dtype,
            )
            # 8. Prepare reference latent variables
            ref_image_latents = self.prepare_ref_latents(
                ref_image,
                batch_size * num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance=False,
            )
            # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            # 10. converting the handle points and target points into disk map

            # 12. pre-compute features of reference net
            # if doing classifier free guidance,
            # we gonna have to repeat the latents at batch dimension
            appr_encoder_hidden_state = negative_prompt_embeds[:, :negative_prompt_embeds.shape[1] - self.num_ip_tokens, :] # only use text part
            if do_classifier_free_guidance:
                ref_image_latents = torch.cat([ref_image_latents] * 2, dim=0)
                appr_encoder_hidden_state = torch.cat([appr_encoder_hidden_state] * 2, dim=0)
                prompt_embeds = torch.cat([prompt_embeds] * 2, dim=0)
            
            self.appearance_encoder(
                ref_image_latents,
                0,
                encoder_hidden_states=appr_encoder_hidden_state,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )
            reference_control_reader.update(reference_control_writer)
            
        latents = self.vae.encode(render_image.to(dtype=self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        with torch.no_grad():
            t_random = torch.randint(int(num_train_timesteps * 0), int(num_train_timesteps * 1), (1,), dtype=torch.long, device=self.device)
            noise = randn_tensor(
                latents.shape, generator=generator, device=device, dtype=latents.dtype
            )
            latents_noisy = self.scheduler.add_noise(latents, noise, t_random)

            latent_model_input = latents_noisy
            
            #tgt inpainting
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_random)
            # latent_model_input = torch.cat([latent_model_input, mask_all_one, masked_image_latents_all_one], dim=1)
            # latents_noisy = latent_model_input
            #region
            # latents_init = self.vae.encode(ref_image.to(dtype=self.vae.dtype)).latent_dist.sample()
            # latents_init = latents_init * self.vae.config.scaling_factor
            # latents_noisy_init = self.scheduler.add_noise(latents_init, noise, t_random)
            #endregion
            
        with torch.enable_grad(),torch.cuda.amp.autocast(enabled=True):
            noise_pred_learnable = self.unet_lora(
                     latents_noisy,
                     t_random,
                     encoder_hidden_states=self.learnable_embeddings.repeat(batch_size,1,1),
                 ).sample
            loss_embedding = F.mse_loss(noise_pred_learnable,noise, reduction="mean")
        
        #region
        # handle_embeddings_list = []
        # target_embeddings_list = []

        # for j in range(len(self.handle_embeddings_list[0])):
        #     handle_embeddings_list.append(torch.cat([emb[j] for emb in [self.handle_embeddings_list[i] for i in camera_idx]],dim=0))
        #     target_embeddings_list.append(torch.cat([emb[j] for emb in [self.target_embeddings_list[i] for i in camera_idx]],dim=0))
        #endregion
        
        #prepare point embeddings
        handle_embeddings_list,target_embeddings_list = self.prepare_point_embeddings_per_batch(handle_points_pixel_list,target_points_pixel_list,height,width)
        
        if do_classifier_free_guidance:
            handle_embeddings_list = [torch.cat([torch.zeros_like(emb), emb], dim=0)
                                      for emb in handle_embeddings_list]
            target_embeddings_list = [torch.cat([torch.zeros_like(emb), emb], dim=0)
                                      for emb in target_embeddings_list]     

        for module in self.attn_modules:
            module.handle_embeddings = handle_embeddings_list
            module.target_embeddings = target_embeddings_list

        with torch.no_grad():
            t = time_step
            noise = randn_tensor(
                latents.shape, generator=generator, device=device, dtype=latents.dtype
            )
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = latents_noisy
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            #tgt inpainting 
            #latent_model_input_all_one = torch.cat([latent_model_input, mask_all_one, masked_image_latents_all_one], dim=1)


            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            # repeat the latent and pad zeros for handle and target embeddings
            if do_classifier_free_guidance:
                latent_model_input = latent_model_input.repeat(2, 1, 1, 1)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            
            noise_pred = noise_pred_uncond + guidance_scale_points * (
                noise_pred_cond - noise_pred_uncond
            )

            with torch.cuda.amp.autocast(enabled=True):
                noise_pred_learnable = self.unet_lora(
                    latents_noisy,
                    t,
                    encoder_hidden_states=self.learnable_embeddings.repeat(batch_size,1,1),
                ).sample
                # tgt inpainting
                # noise_pred_learnable = self.unet_lora(
                #     latent_model_input_all_one,
                #     t,
                #     encoder_hidden_states=self.learnable_embeddings.repeat(batch_size,1,1),
                # ).sample

            w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)  
            alpha = self.alphas_cumprod[t].view(-1, 1, 1, 1) ** 0.5
            sigma = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1) ** 0.5
            
            latents_pred = (latents_noisy - sigma * (noise_pred - noise_pred_learnable + noise)) / alpha
            images_pred = self.vae.decode(latents_pred.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
            images_pred = (images_pred / 2 + 0.5).clamp(0,1)
            
        loss_latent_sds = (F.mse_loss(latents, latents_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / batch_size
        loss_image_sds = (F.mse_loss((render_image/2 + 0.5).to(images_pred.dtype), images_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / batch_size

        return loss_latent_sds, loss_image_sds, loss_embedding


    def compute_sdspp_loss_2d(
        self,
        prompt: Union[str, List[str]] = None,
        ref_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        render_latent: torch.FloatTensor = None,
        mask_image: PIL.Image.Image = None,
        camera_idx = None,
        time_step = None,
        height: int = 512,
        width: int = 512,
        num_train_timesteps: int = 1000,
        guidance_scale_points: float = 3.0,
        guidance_scale_decay: str = 'none',
        num_guidance_steps: int = 50,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        handle_points_pixel_list = None,
        target_points_pixel_list = None,
        skip_cfg_appearance_encoder: bool = False
    ):
        import time
        time_1 = time.time()

        assert self.fusion_blocks in ["midup", "full", "up"]
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale_points > 1.0
        with torch.no_grad():
            reference_control_writer = ReferenceAttentionControl(
                                            self.appearance_encoder,
                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                            skip_cfg_appearance_encoder=skip_cfg_appearance_encoder,
                                            num_images_per_prompt=num_images_per_prompt,
                                            mode='write',
                                            fusion_blocks=self.fusion_blocks,
                                            batch_size = batch_size)
            reference_control_reader = ReferenceAttentionControl(
                                            self.unet,
                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                            skip_cfg_appearance_encoder=skip_cfg_appearance_encoder,
                                            num_images_per_prompt=num_images_per_prompt,
                                            mode='read',
                                            fusion_blocks=self.fusion_blocks,
                                            batch_size = batch_size)
            #TODO:代码精简优化
            # 4. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            prompt_embeds_tuple = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )
            prompt_embeds = prompt_embeds_tuple[0]
            negative_prompt_embeds = prompt_embeds_tuple[1]
            # 4.5 Encode input image
            if self.image_encoder is not None:
                # FIXME: make this generalizable!
                #assert ref_image.shape[0] == 1
                #pil_image = rearrange(ref_image[0], 'c h w -> h w c')
                clip_image_embeds_list = []
                for j in range(batch_size):
                    pil_image = rearrange(ref_image[j], 'c h w -> h w c')
                    pil_image = 127.5 * (pil_image + 1)
                    pil_image = [PIL.Image.fromarray(np.uint8(pil_image.detach().cpu()))]
                    clip_image = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values[0].unsqueeze(0)
                    clip_image_embeds = self.image_encoder(clip_image.to(prompt_embeds.dtype).to(self.image_encoder.device)).image_embeds.unsqueeze(1)
                    clip_image_embeds_list.append(clip_image_embeds)
                clip_image_embeds = torch.cat(clip_image_embeds_list,dim=0)
                # image proj model from IP-Adapter
                image_prompt_embeds = self.image_proj_model(clip_image_embeds)
                negative_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
                # repeat to satisfy the batch size
                image_prompt_embeds = image_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
                negative_image_prompt_embeds = negative_image_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
                #NOTE for LoRA
                prompt_embeds = torch.cat([prompt_embeds_tuple[0], image_prompt_embeds], dim=1)
                negative_prompt_embeds = torch.cat([prompt_embeds_tuple[1], negative_image_prompt_embeds], dim=1)
         

            mask_condition = self.mask_processor.preprocess(
                mask_image, height=height, width=width, resize_mode="default", crops_coords=None
            )
            masked_image = ref_image * (mask_condition.cuda() < 0.5)
            mask, masked_image_latents = self.prepare_mask_latents(
                mask_condition,
                masked_image,
                batch_size * num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance=False,
            )
            # 5. Preprocess reference image
            #NOTE: [-1,1] for image preprocess
            ref_image = self.prepare_image(
                image=ref_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=prompt_embeds.dtype,
            )
            # 8. Prepare reference latent variables
            ref_image_latents = self.prepare_ref_latents(
                ref_image,
                batch_size * num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance=False,
            )
            # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            #extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
            # 10. converting the handle points and target points into disk map

            # 12. pre-compute features of reference net
            # if doing classifier free guidance,
            # we gonna have to repeat the latents at batch dimension
            appr_encoder_hidden_state = negative_prompt_embeds[:, :negative_prompt_embeds.shape[1] - self.num_ip_tokens, :] # only use text part
            if do_classifier_free_guidance:
                ref_image_latents = torch.cat([ref_image_latents] * 2, dim=0)
                appr_encoder_hidden_state = torch.cat([appr_encoder_hidden_state] * 2, dim=0)
                prompt_embeds = torch.cat([prompt_embeds] * 2, dim=0)
            
            self.appearance_encoder(
                ref_image_latents,
                0,
                encoder_hidden_states=appr_encoder_hidden_state,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )
            reference_control_reader.update(reference_control_writer)

        # 13. Denoising loop
        #num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        time_2 = time.time()
            
        latents = render_latent

        with torch.no_grad():
            t_random = torch.randint(int(num_train_timesteps * 0), int(num_train_timesteps * 1), (1,), dtype=torch.long, device=self.device)
            noise = randn_tensor(
                latents.shape, generator=generator, device=device, dtype=latents.dtype
            )
            latents_noisy = self.scheduler.add_noise(latents, noise, t_random)
            #region
            # latents_init = self.vae.encode(ref_image.to(dtype=self.vae.dtype)).latent_dist.sample()
            # latents_init = latents_init * self.vae.config.scaling_factor
            # latents_noisy_init = self.scheduler.add_noise(latents_init, noise, t_random)
            #endregion
            
        with torch.enable_grad(),torch.cuda.amp.autocast(enabled=True):
            noise_pred_learnable = self.unet_lora(
                     latents_noisy,
                     t_random,
                     encoder_hidden_states=self.learnable_embeddings.repeat(batch_size,1,1),
                 ).sample
            loss_embedding = F.mse_loss(noise_pred_learnable,noise, reduction="mean")
        
        #region
        # handle_embeddings_list = []
        # target_embeddings_list = []

        # for j in range(len(self.handle_embeddings_list[0])):
        #     handle_embeddings_list.append(torch.cat([emb[j] for emb in [self.handle_embeddings_list[i] for i in camera_idx]],dim=0))
        #     target_embeddings_list.append(torch.cat([emb[j] for emb in [self.target_embeddings_list[i] for i in camera_idx]],dim=0))
        #endregion
        
        #prepare point embeddings
        handle_embeddings_list,target_embeddings_list = self.prepare_point_embeddings_per_batch(handle_points_pixel_list,target_points_pixel_list,height,width)
        
        if do_classifier_free_guidance:
            handle_embeddings_list = [torch.cat([torch.zeros_like(emb), emb], dim=0)
                                      for emb in handle_embeddings_list]
            target_embeddings_list = [torch.cat([torch.zeros_like(emb), emb], dim=0)
                                      for emb in target_embeddings_list]     

        for module in self.attn_modules:
            module.handle_embeddings = handle_embeddings_list
            module.target_embeddings = target_embeddings_list

        with torch.no_grad():
            t = time_step
            noise = randn_tensor(
                latents.shape, generator=generator, device=device, dtype=latents.dtype
            )
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = latents_noisy
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            # repeat the latent and pad zeros for handle and target embeddings
            if do_classifier_free_guidance:
                latent_model_input = latent_model_input.repeat(2, 1, 1, 1)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            
            noise_pred = noise_pred_uncond + guidance_scale_points * (
                noise_pred_cond - noise_pred_uncond
            )
            #region
            # latents_init = self.vae.encode(ref_image.to(dtype=self.vae.dtype)).latent_dist.sample()
            # latents_init = latents_init * self.vae.config.scaling_factor
            # latents_noisy_init = self.scheduler.add_noise(latents_init, noise, t)
            # latent_model_input_init = self.scheduler.scale_model_input(latents_noisy_init, t)
            # latent_model_input_init = torch.cat([latent_model_input_init, mask, masked_image_latents], dim=1).repeat(2, 1, 1, 1)

            # noise_pred_init = self.unet(
            #         latent_model_input_init,
            #         t,
            #         encoder_hidden_states=prompt_embeds,
            #         cross_attention_kwargs=cross_attention_kwargs,
            #         down_block_additional_residuals=None,
            #         mid_block_additional_residual=None,
            #         return_dict=False,
            #     )[0]
            # noise_pred_uncond_init, noise_pred_cond_init = noise_pred_init.chunk(2)
            # noise_pred_init = noise_pred_uncond_init + guidance_scale_points * (
            #     noise_pred_cond_init - noise_pred_uncond_init
            # )
            #endregion

            with torch.cuda.amp.autocast(enabled=True):
                noise_pred_learnable = self.unet_lora(
                    latents_noisy,
                    t,
                    encoder_hidden_states=self.learnable_embeddings.repeat(batch_size,1,1),
                ).sample

            w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)  
            alpha = self.alphas_cumprod[t].view(-1, 1, 1, 1) ** 0.5
            sigma = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1) ** 0.5


            latents_pred = (latents_noisy - sigma * (noise_pred - noise_pred_learnable + noise)) / alpha
            images_pred = self.vae.decode(latents_pred.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
            images_pred = (images_pred / 2 + 0.5).clamp(0,1)
            
        loss_latent_sds = (F.mse_loss(latents, latents_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / batch_size
        
        render_image = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
        loss_image_sds = (F.mse_loss((render_image/2 + 0.5).to(images_pred.dtype), images_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / batch_size
        #loss_image_sds = (F.mse_loss(render_image.to(images_pred.dtype), images_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / batch_size
        time_3 = time.time()

        #print(f"{(time_2 - time_1)/(time_3 - time_1):.4f}")
        return loss_latent_sds, loss_image_sds, loss_embedding
    
    @torch.no_grad()
    def denoise_2d(
        self,
        prompt: Union[str, List[str]] = None,
        ref_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        render_latent: torch.FloatTensor = None,
        mask_image: PIL.Image.Image = None,
        camera_idx = None,
        height: int = 512,
        width: int = 512,
        num_train_timesteps: int = 1000,
        guidance_scale_points: float = 3.0,
        num_inference_steps: int = 8,
        guidance_scale_decay: str = 'none',
        num_guidance_steps: int = 50,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        handle_points_pixel_list = None,
        target_points_pixel_list = None,
        skip_cfg_appearance_encoder: bool = False,
    ):
        import time
        time_1 = time.time()

        assert self.fusion_blocks in ["midup", "full", "up"]
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale_points > 1.0
        with torch.no_grad():
            reference_control_writer = ReferenceAttentionControl(
                                            self.appearance_encoder,
                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                            skip_cfg_appearance_encoder=skip_cfg_appearance_encoder,
                                            num_images_per_prompt=num_images_per_prompt,
                                            mode='write',
                                            fusion_blocks=self.fusion_blocks,
                                            batch_size = batch_size)
            reference_control_reader = ReferenceAttentionControl(
                                            self.unet,
                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                            skip_cfg_appearance_encoder=skip_cfg_appearance_encoder,
                                            num_images_per_prompt=num_images_per_prompt,
                                            mode='read',
                                            fusion_blocks=self.fusion_blocks,
                                            batch_size = batch_size)
            #TODO:代码精简优化
            # 4. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            prompt_embeds_tuple = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )
            prompt_embeds = prompt_embeds_tuple[0]
            negative_prompt_embeds = prompt_embeds_tuple[1]
            # 4.5 Encode input image
            if self.image_encoder is not None:
                # FIXME: make this generalizable!
                #assert ref_image.shape[0] == 1
                #pil_image = rearrange(ref_image[0], 'c h w -> h w c')
                clip_image_embeds_list = []
                for j in range(batch_size):
                    pil_image = rearrange(ref_image[j], 'c h w -> h w c')
                    pil_image = 127.5 * (pil_image + 1)
                    pil_image = [PIL.Image.fromarray(np.uint8(pil_image.detach().cpu()))]
                    clip_image = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values[0].unsqueeze(0)
                    clip_image_embeds = self.image_encoder(clip_image.to(prompt_embeds.dtype).to(self.image_encoder.device)).image_embeds.unsqueeze(1)
                    clip_image_embeds_list.append(clip_image_embeds)
                clip_image_embeds = torch.cat(clip_image_embeds_list,dim=0)
                # image proj model from IP-Adapter
                image_prompt_embeds = self.image_proj_model(clip_image_embeds)
                negative_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
                # repeat to satisfy the batch size
                image_prompt_embeds = image_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
                negative_image_prompt_embeds = negative_image_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
                #NOTE for LoRA
                prompt_embeds = torch.cat([prompt_embeds_tuple[0], image_prompt_embeds], dim=1)
                negative_prompt_embeds = torch.cat([prompt_embeds_tuple[1], negative_image_prompt_embeds], dim=1)
         

            mask_condition = self.mask_processor.preprocess(
                mask_image, height=height, width=width, resize_mode="default", crops_coords=None
            )
            masked_image = ref_image * (mask_condition.cuda() < 0.5)
            mask, masked_image_latents = self.prepare_mask_latents(
                mask_condition,
                masked_image,
                batch_size * num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance=False,
            )
            # 5. Preprocess reference image
            #NOTE: [-1,1] for image preprocess
            ref_image = self.prepare_image(
                image=ref_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=prompt_embeds.dtype,
            )
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            # 8. Prepare reference latent variables
            ref_image_latents = self.prepare_ref_latents(
                ref_image,
                batch_size * num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance=False,
            )
            # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
            # 10. converting the handle points and target points into disk map

            # 12. pre-compute features of reference net
            # if doing classifier free guidance,
            # we gonna have to repeat the latents at batch dimension
            appr_encoder_hidden_state = negative_prompt_embeds[:, :negative_prompt_embeds.shape[1] - self.num_ip_tokens, :] # only use text part
            if do_classifier_free_guidance:
                ref_image_latents = torch.cat([ref_image_latents] * 2, dim=0)
                appr_encoder_hidden_state = torch.cat([appr_encoder_hidden_state] * 2, dim=0)
                prompt_embeds = torch.cat([prompt_embeds] * 2, dim=0)
            
            self.appearance_encoder(
                ref_image_latents,
                0,
                encoder_hidden_states=appr_encoder_hidden_state,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )
            reference_control_reader.update(reference_control_writer)

        # 13. Denoising loop
        #num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        time_2 = time.time()
            
        latents = render_latent

        #prepare point embeddings
        handle_embeddings_list,target_embeddings_list = self.prepare_point_embeddings_per_batch(handle_points_pixel_list,target_points_pixel_list,height,width)
        
        if do_classifier_free_guidance:
            handle_embeddings_list = [torch.cat([torch.zeros_like(emb), emb], dim=0)
                                      for emb in handle_embeddings_list]
            target_embeddings_list = [torch.cat([torch.zeros_like(emb), emb], dim=0)
                                      for emb in target_embeddings_list]     

        for module in self.attn_modules:
            module.handle_embeddings = handle_embeddings_list
            module.target_embeddings = target_embeddings_list

        noise = randn_tensor(
            latents.shape, generator=generator, device=device, dtype=latents.dtype
        )
        latents_noisy = self.scheduler.add_noise(latents, noise, torch.tensor([999]))
        # repeat the latent and pad zeros for handle and target embeddings
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        latents_noisy_750 = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t < 750 and latents_noisy_750 is None:
                    latents_noisy_750 = latents_noisy
                    alpha = self.alphas_cumprod[t].view(-1, 1, 1, 1) ** 0.5
                    sigma = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1) ** 0.5
                    latent_model_input = latents_noisy_750
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                    # repeat the latent and pad zeros for handle and target embeddings
                    if do_classifier_free_guidance:
                        latent_model_input = latent_model_input.repeat(2, 1, 1, 1)
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=None,
                        mid_block_additional_residual=None,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    cur_guidance_scale_points = (guidance_scale_points-1.0) * (1.0 - i/len(timesteps)) ** 2 + 1
                    noise_pred = noise_pred_uncond + cur_guidance_scale_points * (noise_pred_cond - noise_pred_uncond)
                    latents_750_pred_x0 = (latents_noisy_750 - sigma * noise_pred) / alpha

                latent_model_input = latents_noisy
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # repeat the latent and pad zeros for handle and target embeddings
                if do_classifier_free_guidance:
                    latent_model_input = latent_model_input.repeat(2, 1, 1, 1)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=None,
                    mid_block_additional_residual=None,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                guidance_scale_decay = "inv_square"
                # perform guidance
                if guidance_scale_decay == "none":
                    cur_guidance_scale_points = guidance_scale_points
                elif guidance_scale_decay == "linear":
                    cur_guidance_scale_points = \
                        guidance_scale_points * (1.0 - i / len(timesteps)) + \
                        1.0 * i / len(timesteps)
                elif guidance_scale_decay == "square":
                    cur_guidance_scale_points = \
                        guidance_scale_points * (1.0 - (i / len(timesteps)) ** 2) + \
                        1.0 * (i / len(timesteps)) ** 2
                elif guidance_scale_decay == "quadratic":
                    cur_guidance_scale_points = \
                        guidance_scale_points * (1.0 - (i / len(timesteps)) ** 3) + \
                        1.0 * (i / len(timesteps)) ** 3
                elif guidance_scale_decay == "inv_square":
                    cur_guidance_scale_points = \
                        (guidance_scale_points-1.0) * (1.0 - i/len(timesteps)) ** 2 + 1
                else:
                    raise NotImplementedError("decay schedule not implemented")

                # only perform guidance on the
                # first "num_guidance_steps" denoising steps
                if num_guidance_steps is None or \
                    (num_guidance_steps is not None and i < num_guidance_steps):
                    noise_pred = noise_pred_uncond + \
                        cur_guidance_scale_points * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                # compute the previous noisy sample x_t -> x_t-1
                latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        #print(f"{(time_2 - time_1)/(time_3 - time_1):.4f}")
        return latents_noisy,latents_noisy_750,latents_750_pred_x0