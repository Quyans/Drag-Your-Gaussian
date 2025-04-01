import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from einops import rearrange

from diffusers import StableDiffusionPipeline
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from transformers import PretrainedConfig

# override unet forward
# The only difference from diffusers:
# return intermediate UNet features of all UpSample blocks

# obtain the bilinear interpolated feature patch centered around (x, y) with radius r

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def get_generator(seed, device):
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
