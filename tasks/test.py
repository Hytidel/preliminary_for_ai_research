from utils.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from omegaconf import OmegaConf, DictConfig

from pathlib import Path

import torch

import gc

from tqdm.auto import tqdm

from utils.basic_utils import get_global_variable, is_none, get_true_value
from utils.image_utils import load_img_path, save_pil_as_png
from utils.yaml_utils import (
    load_yaml, save_yaml, 
    convert_numpy_type_to_native_type
)


def test_test(
    cfg: DictConfig
) -> List[float]:
    logger(
        f"Hello"
    )

    import torch
    
    print(
        torch.cuda.is_available()
    )

    import pickle

    breakpoint()

    # `test_test()` done
    pass





def test_implement(
    cfg: DictConfig
):
    # ---------= [Global Variables] =---------
    logger(f"[Global Variables] Loading started. ")

    exp_name = get_global_variable("exp_name")
    device = get_global_variable("device")
    seed = get_global_variable("seed")

    logger(f"[Global Variables] Loading finished. ")

    # ---------= [Task] =---------
    test_test(cfg)

    # test_dinosaur(cfg)

    # `test_implement()` done
    pass

def test(
    cfg: DictConfig
):
    test_implement(cfg)

    pass