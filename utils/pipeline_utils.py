from utils.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import numpy as np

import torch
import torchvision

from PIL import Image

from pathlib import Path

import re

import gc

from utils.basic_utils import get_attr


INFERENCE_STEP_MINUS_ONE_SCHEDULER_LIST = [
    "PNDMScheduler"
]


def load_pipeline(
    pipeline_type: str, 
    pipeline_path: str, 
    torch_dtype: str = None, 
    variant: str = None, 
):
    pipeline = get_attr("diffusers.pipelines", pipeline_type) \
        .from_pretrained(
            pipeline_path, 
            torch_dtype = getattr(torch, torch_dtype), 
            variant = variant
        )

    pipeline.inv_scheduler = None

    return pipeline


def load_scheduler(
    scheduler_type: str, 
    pipeline
):
    scheduler = get_attr("diffusers.schedulers", scheduler_type) \
        .from_config(pipeline.scheduler.config)

    return scheduler


def get_inference_step_minus_one(
    scheduler_type: str
) -> bool:
    return scheduler_type in INFERENCE_STEP_MINUS_ONE_SCHEDULER_LIST


def load_unet(
    unet_type: str,  # ["UNet2DModel", "UNet2DConditionModel"]
    pipeline
):
    unet = get_attr("diffusers.models", unet_type) \
        .from_config(pipeline.unet.config)
        
    return unet


@torch.no_grad()
def img_pil_to_latent(
    img_pil, 
    pipeline
) -> torch.Tensor:
    # value [0, 255] -> [0, 1] -> [-1, 1]
    img_tensor = torchvision.transforms.functional.to_tensor(img_pil) \
        .unsqueeze(0) * 2 - 1

    img_tensor = img_tensor.to(
        dtype = pipeline.vae.dtype, 
        device = pipeline.device
    )
    
    latent = pipeline.vae.encode(img_tensor)
    
    latent = pipeline.vae.config.scaling_factor * latent.latent_dist.sample()

    return latent


@torch.no_grad()
def img_latent_to_pil(
    # img_latent.shape = (num_latent, latent_num_channel, latent_height, latent_width)
    img_latent_list, 
    pipeline, 

    batch_size: Optional[int] = 20
) -> Image.Image:
    if not isinstance(img_latent_list, torch.Tensor):
        img_latent_list = torch.tensor(
            img_latent_list, 

            dtype = img_latent_list[0].dtype, 
            device = img_latent_list[0].device
        )

    num_latent = img_latent_list.shape[0]

    if hasattr(pipeline, "decode_latents"):
        img_numpy_list = []

        for i in range(0, num_latent, batch_size):
            batch_img_latent_list = img_latent_list[i: min(i + batch_size, num_latent)]
            batch_img_numpy_list = pipeline.decode_latents(batch_img_latent_list)

            img_numpy_list.append(batch_img_numpy_list)

            # ---------= [Clean Up] =----------
            del batch_img_latent_list
            gc.collect()
            torch.cuda.empty_cache()
            
            # goto `for i`
            pass

        # img_numpy_list.shape = (num_img, height, width, num_channel)
        img_numpy_list = np.vstack(img_numpy_list)

        img_numpy_list = np.nan_to_num(
            img_numpy_list, 
            nan = 0.0, 
            neginf = 0.0, posinf = 1.0
        )

        img_numpy_list = np.clip(
            img_numpy_list, 
            0.0, 1.0
        )

        img_pil_list = pipeline.numpy_to_pil(img_numpy_list)
    else:
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
        
        # make sure the VAE is in `float32` mode, as it overflows in `float16`
        if (pipeline.vae.dtype == torch.float16) and pipeline.vae.config.force_upcast:
            pipeline.upcast_vae()
            img_latent_list = img_latent_list.to(
                next(
                    iter(
                        pipeline.vae.post_quant_conv.parameters()
                    )
                ).dtype
            )
        
        img_pil_list = []
        
        for i in range(0, num_latent, batch_size):
            batch_img_latent_list = img_latent_list[i: min(i + batch_size, num_latent)]

            batch_img_latent_list = batch_img_latent_list / pipeline.vae.config.scaling_factor

            if isinstance(pipeline, StableDiffusion3Pipeline):
                batch_img_latent_list += pipeline.vae.config.shift_factor

            batch_img_tensor_list = pipeline.vae.decode(
                batch_img_latent_list, 
                return_dict = False
            )[0]

            batch_img_pil_list = pipeline.image_processor.postprocess(
                batch_img_tensor_list, 
                output_type = "pil"
            )

            img_pil_list += batch_img_pil_list

            # ---------= [Clean Up] =----------
            del batch_img_latent_list
            del batch_img_tensor_list
            gc.collect()
            torch.cuda.empty_cache()
            
            # goto `for i`
            pass

    # ---------= [Clean Up] =----------
    del img_latent_list
    gc.collect()
    torch.cuda.empty_cache()

    return img_pil_list


def process_prompt_list(
    prompt: Union[str, List[str]], 
    batch_size: Optional[int] = None, 
    negative_prompt: Union[str, List[str]] = None, 
):
    if isinstance(prompt, list):
        if batch_size is None:
            batch_size = len(prompt)
        elif batch_size != len(prompt):
            raise ValueError(
                f"The length of the `prompt` list doesn't match `batch_size`, "
                f"got {len(prompt)} and {batch_size}. "
            )
    elif batch_size is None:
        batch_size = 1
        prompt = [prompt]
    else:
        prompt = [prompt] * batch_size

    if negative_prompt is not None:
        if isinstance(negative_prompt, list):
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"The length of the `negative_prompt` list doesn't match `batch_size`, "
                    f"got {len(negative_prompt)} and {batch_size}. "
                    )
        else:
            negative_prompt = [negative_prompt] * batch_size

    return prompt, batch_size, negative_prompt


def preprocess_prompt(
    prompt: str
) -> str:
    res_prompt = prompt.strip('"')

    res_prompt = re.sub(
        r"[^a-zA-Z0-9,. ]", '', 
        res_prompt
    )

    return res_prompt


def get_folder_name(
    prompt: str, 
    used_folder_name_list: List[str]
) -> str:
    """
    NB:
        The derived folder name will be inserted to `used_folder_name_list`. 

    Func:
        Derive the folder name corresponding to the prompt by concatenating the initial words 
            of the prompt to form a unique unused folder name. 
    
    Ret:
        `folder_name` (`str`): The derived folder name. 
    """

    prompt = preprocess_prompt(prompt)
    word_list = prompt.split()

    num_word = 4
    while num_word < len(word_list):
        folder_name = word_list[: num_word]
        folder_name = '_'.join(
            [word.lower() for word in folder_name]
        )

        if folder_name not in used_folder_name_list:
            used_folder_name_list.append(folder_name)

            return folder_name
        else:
            num_word += 1

    folder_name = '_'.join(
        [word.lower() for word in word_list]
    )
    used_folder_name_list.append(folder_name)

    return folder_name


def get_pipeline_category_and_type(
    pipeline_path: Union[str, Path]
) -> Tuple[str, str]:
    """
    Func:
        Get the category and type of the pipeline from `pipeline_path`. 

    Ret:
        (`pipeline_category_name`, `pipeline_type_name`) (`Tuple[str, str]`): 
            `pipeline_category_name` is the name of the category where the pipeline belongs to, 
            and `pipeline_type_name` is the type of the pipeline. 
    """

    if isinstance(pipeline_path, str):
        pipeline_path = Path(pipeline_path)
    
    folder_name = pipeline_path.name

    res = None

    if folder_name == "sd-turbo":
        res = (
            "sd_family", 
            "sd-turbo"
        )
    elif folder_name == "stable-diffusion-v1-4":
        res = (
            "sd_family", 
            "sd_v1_4"
        )
    elif folder_name == "stable-diffusion-2-1-base":
        res = (
            "sd_family", 
            "sd_v2_1"
        )

    elif folder_name == "sdxl-turbo":
        res = (
            "sdxl_family", 
            "sdxl-turbo"
        )
    elif folder_name == "stable-diffusion-xl-base-1.0":
        res = (
            "sdxl_family", 
            "sdxl"
        )
    
    elif folder_name == "stable-diffusion-3.5-medium":
        res = (
            "sd_3_family", 
            "sd_v3_5_medium"
        )

    elif folder_name == "hunyuan_dit_v1_2":
        res = (
            "hunyuan_dit_family", 
            "hunyuan_dit_v1_2"
        )
    
    elif folder_name == "PixArt-XL-2-1024-MS":
        res = (
            "pixart_alpha_family", 
            "pixart_alpha_xl"
        )

    else:
        raise ValueError(
            f"Unsupported `folder_name`, got `{folder_name}`. "
        )

    # `get_pipeline_type()` done
    return res
