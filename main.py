from utils.logger import logger

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.basic_utils import (
    set_global_variable_dict, 
    get_global_variable, set_global_variable
)


def test(
    cfg: DictConfig
):
    from tasks.test import test
    test(cfg)   

    # `test()` done
    pass





def run_task(
    cfg: DictConfig
):
    task_name = cfg["task"]["name"]
    
    if task_name.startswith("test"):
        test(cfg)
    # elif task_name.startswith("do_ddim_inversion"):
    #     do_ddim_inversion(cfg)

    else:
        raise NotImplementedError(
            f"Unsupported task: `{task_name}`. "
        )


@hydra.main(version_base = None, config_path = "config", config_name = "cfg")
def main(
    cfg: DictConfig
):
    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(
        cfg, 
        resolve = True
    )

    set_global_variable_dict(cfg)

    exp_name = get_global_variable("exp_name")
    logger(f"Start experiment `{exp_name}`. ")

    run_task(cfg)

    logger(f"Experiment `{exp_name}` finished. ")

    # `main()` done
    pass


if __name__ == "__main__":
    main()
    