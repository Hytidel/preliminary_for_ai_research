from utils.logger import logger

from typing import Any, Union, Tuple, List, Optional

import pandas as pd

from pathlib import Path


def load_csv(
    csv_path: Union[str, Path], 

    delimiter: Optional[str] = ','
) -> pd.DataFrame:
    if isinstance(csv_path, str):
        csv_path = Path(csv_path)

    data_frame = pd.read_csv(
        csv_path, 

        delimiter = delimiter
    )

    # `load_csv()` done
    return data_frame


def get_shape(
    data_frame: pd.DataFrame
) -> Tuple:
    shape = data_frame.shape

    # `get_shape()` done
    return shape


def get_row_list(
    data_frame: pd.DataFrame
) -> List[Tuple[int, pd.Series]]:
    row_list = [
        row \
            for row_idx, row in data_frame.iterrows()
    ]

    # `get_row_list()` done
    return row_list


def get_col_name_list(
    data_frame: pd.DataFrame
) -> List[Tuple[int, pd.Series]]:
    col_name_list = [
        col_name \
            for col_name in data_frame.columns
    ]

    # `col_name_list()` done
    return col_name_list
    

def get_row(
    data_frame: pd.DataFrame, 
    row_idx: int
) -> pd.Series:
    row = data_frame.loc[row_idx]

    # `get_row()`
    return row


def get_col(
    data_frame: pd.DataFrame, 
    col_name: str
) -> pd.Series:
    col = data_frame[col_name]

    # `get_col()`
    return col


def get_element(
    data_frame: pd.DataFrame, 
    row_idx: int, 
    col_name: str
) -> Any:
    element = data_frame.loc[row_idx, col_name]

    # `get_element()`
    return element
