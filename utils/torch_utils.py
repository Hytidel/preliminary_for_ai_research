from utils.logger import logger

from typing import Any, Optional, Tuple, Union, Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from contextlib import contextmanager

from pathlib import Path


def tsfm_to_1d_array(
    array: Union[Any, List[Any], torch.Tensor], 
    target_length: int, 

    dtype: str = None, 
    device: str = None
) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        if array.ndim >= 2:
            raise ValueError(
                f"Got a 2D `array`, {array}. "
            )
        
        length = len(array)
        if length == 1:
            array = torch.tile(
                array, 
                (target_length, )
            )
        elif length != target_length:
            raise ValueError(
                f"The length of `array` ({length}) does not match `target_length` ({target_length}). "
            )
        
        return array
    elif isinstance(array, (int, float, bool)):
        if (dtype is None) or (device is None):
            raise ValueError(
                f"`dtype` and `device` should be specified if `array` is a scalar. "
            )

        array = torch.tensor(
            [array] * target_length, 

            dtype = dtype, 
            device = device
        )
        
        return array
    elif isinstance(array, list):
        if isinstance(array[0], list):
            raise ValueError(
                f"Got a 2D `array`, {array}. "
            )
        
        length = len(array)
        if length == 1:
            array = [array[0]] * target_length
        elif length != target_length:
            raise ValueError(
                f"The length of `array` ({length}) does not match `target_length` ({target_length}). "
            )
        
        for element in array:
            if isinstance(element, torch.Tensor):
                dtype = element.dtype
                device = element.device

                break

        if (dtype is None) or (device is None):
            raise ValueError(
                f"`dtype` and `device` should be specified if `array[0]` is a scalar. "
            )

        array = torch.tensor(
            array, 

            dtype = dtype, 
            device = device
        )

        return array
    else:
        raise NotImplementedError(
            f"Unsupported type of `array`, got {type(array)}. "
        )
    

def get_device():
    device = "cuda" if torch.cuda.is_available() \
        else "cpu"

    return device


def get_optim(
    optim_type: str, 
    model: nn.Module, 
    lr: float, 

    # AdamW
    adamw_beta_tuple: Optional[Tuple[float, float]] = (0.9, 0.99)
):
    if optim_type == "Adam":
        optim = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()), 
            lr = lr
        )
    elif optim_type == "AdamW":
        optim = torch.optim.AdamW(
            filter(lambda param: param.requires_grad, model.parameters()), 
            lr = lr, 

            betas = adamw_beta_tuple
        )
    else:
        raise NotImplementedError(
            f"Unsupported optimizer:` {optim_type}`. "
        )

    return optim


def get_lr_scheduler(
    lr_scheduler_type: str, 
    optim, 

    # `ReduceLROnPlateau` param
    mode: str = "min", 
    factor: float = None, 
    patience: int = None, 
    cooldown: int = None, 
    threshold: float = None, 

    verbose: bool = True, 
):
    """
    Args:
        verbose (`bool`, *optional*, defaults to True):
            Set `verbose = True` for `lr_scheduler` to print prompt messages 
            when the learning rate changes. 
    """

    if lr_scheduler_type is None:
        lr_scheduler = None
    elif lr_scheduler_type == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            mode = mode, 
            factor = factor, 
            patience = patience, 
            cooldown = cooldown, 
            threshold = threshold, 

            verbose = verbose
        )
    else:
        raise NotImplementedError(
            f"Unsupported learning rate scheduler:` {lr_scheduler_type}`. "
        )
    
    return lr_scheduler


def get_criterion(
    criterion_type
):
    if criterion_type == "L1":
        criterion = F.l1_loss
    elif criterion_type in ["L2", "MSE"]:
        criterion = F.mse_loss
    elif criterion_type == "Huber":
        criterion = F.smooth_l1_loss
    else:
        raise NotImplementedError(
            f"Unsupported criterion:` {criterion_type}`. "
        )

    return criterion


def save_model_state_dict(
    state_dict: Dict, 

    ckpt_root_path: Union[str, Path], 
    ckpt_filename: str
):
    if isinstance(ckpt_root_path, str):
        ckpt_root_path = Path(ckpt_root_path)
    
    ckpt_root_path.mkdir(parents = True, exist_ok = True)
    
    ckpt_path = ckpt_root_path / ckpt_filename

    torch.save(
        state_dict, 
        ckpt_path
    )


def save_model_ckpt(
    model, 

    ckpt_root_path: str, 
    ckpt_filename: str
):
    state_dict = model.state_dict()

    save_model_state_dict(
        state_dict, 
        ckpt_root_path, 
        ckpt_filename
    )


def load_model_state_dict(
    state_dict_path: str, 
    device: str
) -> Dict:
    state_dict_path = Path(state_dict_path)
    
    if (state_dict_path is None) or (state_dict_path == "None") \
        or (not state_dict_path.is_file()):
        logger(
            f"State dict `{state_dict_path}` not exists, continue with initial model parameters. ", 
            log_type = "warning"
        )

        return None
    
    state_dict = torch.load(
        state_dict_path, 
        map_location = device
    )

    return state_dict


def load_model_ckpt(
    model, 
    ckpt_path: str, 
    device: str, 
    strict: Optional[bool] = False
):
    ckpt_path = Path(ckpt_path)

    if (ckpt_path is None) or (ckpt_path == "None") \
        or (not ckpt_path.is_file()):
            logger(
                f"Model checkpoint `{ckpt_path}` not exists, continue with initial model parameters. ", 
                log_type = "info"
            )

            return

    state_dict = torch.load(
        ckpt_path, 
        map_location = device
    )

    model.load_state_dict(
        state_dict, 
        strict = strict
    )

    logger(
        f"Loaded model checkpoint `{ckpt_path}`.", 
        log_type = "info"
    )


def get_model_num_param(
    model
):
    model_num_param = sum(
        [
            param.numel() \
                for param in model.parameters()
        ]
    )

    return model_num_param


def get_current_lr_list(
    optim
):
    cur_lr_list = [
        param_group["lr"] \
            for param_group in optim.param_groups
    ]

    return cur_lr_list


def get_generator(
    seed, 
    device
):
    if isinstance(seed, torch.Tensor):
        seed = int(seed)

    generator = torch.Generator(device) \
        .manual_seed(seed)
    
    return generator


def get_selected_state_dict(
    model, 
    selected_param_name_list: List[str]
) -> Dict:
    state_dict = model.state_dict()

    selected_state_dict = {
        name: state_dict[name] \
            for name in selected_param_name_list
    }

    return selected_state_dict


@contextmanager
def determine_enable_grad(
    enable_grad: bool
):
    if enable_grad:
        with torch.enable_grad():
            yield
    else:
        with torch.no_grad():
            yield


def get_latent(
    shape: Tuple, 

    generator: Optional[torch.Generator] = None, 
    seed: Optional[int] = None, 

    device: Optional[str] = "cpu", 
    dtype: Optional[str] = "float32"
) -> torch.Tensor:
    from diffusers.utils.torch_utils import randn_tensor

    if (generator is not None) and (seed is not None):
        logger(
            f"Both `generator` and `seed` are provided, `generator` prioritzes. ", 
            log_type = "warning"
        )
    
    if (generator is None) and (seed is not None):
        generator = get_generator(
            seed = seed, 
            device = device
        )

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if isinstance(device, str):
        device = torch.device(device)

    latent = randn_tensor(
        shape = shape, 

        generator = generator, 

        device = device, 
        dtype = dtype
    )

    # `get_latent()` done
    return latent


def get_list_slicing(
    tensor_list: Union[torch.Tensor, List[torch.Tensor]], 

    st: int = 0, 
    ed: int = -1
) -> torch.Tensor:
    """
    Func:
        Get a slicing [st: ed] from `tensor_list`. 
        Remind that index `ed` is not included. 

    Ret:
        `tensor_list_slicing` (`torch.Tensor`): The required slicing. 
    """

    if isinstance(tensor_list, list):
        tensor_list_slicing = torch.stack(
            tensor_list[st: ed]
        )
    elif isinstance(tensor_list, torch.Tensor):
        tensor_list_slicing = tensor_list[st: ed]
    else:
        raise ValueError(
            f"Unsupported type of `tensor_list`, got `{type(tensor_list)}`. "
        )

    # `get_list_slicing()` done
    return tensor_list_slicing
