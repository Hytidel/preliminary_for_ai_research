from utils.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from io import BytesIO

from pathlib import Path

import shutil


def copy_file(
    src_path: Union[str, Path], 

    dst_root_path: Union[str, Path], 
    filename: str
):
    if not isinstance(src_path, Path):
        src_path = Path(src_path)
    if not isinstance(dst_root_path, Path):
        dst_root_path = Path(dst_root_path)

    dst_root_path.mkdir(
        parents = True, 
        exist_ok = True
    )

    dst_path = dst_root_path / filename

    shutil.copy(src_path, dst_path)

    # `copy_file()` done
    pass


def copy_directory(
    src_root_path: Union[str, Path], 

    dst_parent_root_path: Union[str, Path], 
    directory_name: str
):
    if not isinstance(src_root_path, Path):
        src_root_path = Path(src_root_path)
    if not isinstance(dst_parent_root_path, Path):
        dst_parent_root_path = Path(dst_parent_root_path)

    dst_parent_root_path.mkdir(
        parents = True, 
        exist_ok = True
    )

    dst_root_path = dst_parent_root_path / directory_name

    if directory_name.is_dir():
        raise FileExistsError(
            f"The directory `{str(dst_root_path)}` already exists. "
        )

    shutil.copytree(src_root_path, dst_root_path)

    # `copy_directory()` done
    pass
