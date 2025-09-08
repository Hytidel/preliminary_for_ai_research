from utils.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol, Any

import numpy as np

import copy


def tsfm_to_1d_array(
    array: Union[Any, List[Any], np.ndarray], 
    target_length: int, 

    dtype: str = None, 
    device: str = None  # useless
) -> np.ndarray:
    if isinstance(array, np.ndarray):
        if array.ndim >= 2:
            raise ValueError(
                f"Got a 2D `array`, {array}. "
            )
        
        length = len(array)
        if length == 1:
            array = np.tile(
                array, 
                target_length
            )
        elif length != target_length:
            raise ValueError(
                f"The length of `array` ({length}) does not match `target_length` ({target_length}). "
            )
        
        return array
    elif isinstance(array, (int, float, bool)):
        if dtype is None:
            raise ValueError(
                f"`dtype` should be specified if `array` is a scalar. "
            )
        
        array = np.asarray(
            [array] * target_length, 

            dtype = dtype
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
            if isinstance(element, np.ndarray):
                dtype = element.dtype

                break
        
        if dtype is None:
            raise ValueError(
                f"`dtype` should be specified if `array` is a scalar. "
            )

        array = np.array(
            array, 

            dtype = dtype
        )

        return array
    else:
        raise NotImplementedError(
            f"Unsupported type of `array`, got {type(array)}. "
        )


# TODO: check
def tsfm_to_2d_matrix(
    matrix: Union[float, List[float], List[List[float]], np.ndarray], 
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    func:
        Transform `matrix` to shape `target_shape` by duplication. 
    """

    num_row, num_col = target_shape

    if isinstance(matrix, float):
        matrix = [matrix] * num_col
        matrix = np.asarray(
            [copy.deepcopy(matrix) for _ in range(num_row)]
        )
    elif isinstance(matrix, list):
        # matrix: List[float]
        if isinstance(matrix[0], float):
            if len(matrix) != num_col:
                raise ValueError(
                    f"The length of `matrix` does not match `num_col`, "
                    f"got {len(matrix)}, but `num_rol = {num_col}`. "
                )
            else:
                matrix = np.asarray(
                    [copy.deepcopy(matrix) for _ in range(num_row)]
                )
        # weight_matrix: List[List[float]]
        elif (len(matrix) != num_row) or (len(matrix[0]) != num_col):
            raise ValueError(
                f"The shape of `matrix` does not match the shape {target_shape}, "
                f"got ({len(matrix)}, {len(matrix[0])}), "
                f"but `target_shape = {target_shape}`. "
            )
    elif isinstance(matrix, np.ndarray):
        if matrix.ndim == 1:
            matrix = np.tile(
                matrix, 
                (num_row, 1)
            )
        
        if matrix.shape != target_shape:
            raise ValueError(
                f"The shape of `matrix` does not match the shape {target_shape}, "
                f"got ({len(matrix)}, {len(matrix[0])}), "
                f"but `target_shape = {target_shape}`. "
            )
    else:
        raise NotImplementedError(
            f"Unsupported type of `matrix`, got {type(matrix)}. "
        )

    # matrix.shape = target_shape
    return matrix
