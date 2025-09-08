from utils.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

from pathlib import Path

import numpy as np

import io

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image


def save_plot(
    fig, 
    plot_root_path: Union[str, Path], 
    plot_filename: str
):
    if not isinstance(plot_root_path, Path):
        plot_root_path = Path(plot_root_path)
    
    plot_root_path.mkdir(
        parents = True, 
        exist_ok = True
    )

    plot_path = plot_root_path / plot_filename
    fig.savefig(plot_path)


def fig_to_pil(
    fig
) -> Image.Image:
    """
    Func: 
        将 `(fig, ax)` 中的 `fig` 转化为 PIL 对象。

    Ret: 
        `fig_pil` (`Image.Image`): `fig` 对应的 PIL 对象。
    """
    
    buf = io.BytesIO()

    fig.tight_layout()

    fig.savefig(
        buf, 

        format = "png", 
        bbox_inches = "tight"
    )

    buf.seek(0)

    fig_pil = Image.open(buf) \
        .convert("RGB")

    # `fig_to_pil()` done
    return fig_pil


def get_line_chart(
    figsize: Optional[Union[float, Tuple[float, float]]] = (10, 5), 

    x_list: Optional[Union[List[float], np.ndarray]] = None, 
    y_list_list: Union[List[List[float]], List[np.ndarray], np.ndarray] = None, 
    y_label_list: Union[str, List[str]] = None, 

    marker_list: Union[str, List[str]] = [
        'o',  # circle
        's',  # square
        '*',  # star
        '+',  # plus
        'x',  # x

        'd',  # diamond
        'D',  # thin diamond

        '^',  # up-pointing triangle
        'v',  # down-pointing triangle
        '<',  # left-pointing triangle
        '>',  # right-pointing triangle

        'p',  # pentagon
        'h',  # hexagon 1
        'H',  # hexagon 2
    ], 
    marker_size_list: Union[int, float, List[int], List[float]] = [6], 

    color_list: Union[str, List[str]] = None,  # color name or hex
    alpha_list: Union[float, List[float]] = None,

    num_sample: int = None, 
    std_list_list: Union[float, List[List[float]], np.ndarray] = None, 
    confidence_level: float = None, 

    fill_alpha_list: Union[float, List[float]] = None, 
    face_color_list: Union[str, List[str]] = None,  # color name or hex

    plot_title: Optional[str] = None, 
    plot_x_label: Optional[str] = None, 
    plot_y_label: Optional[str] = None, 

    show_grid: Optional[bool] = True, 
    show_legend: Optional[bool] = True
) -> Tuple:
    # get `figsize`
    if isinstance(figsize, tuple):
        pass
    elif isinstance(figsize, float):
        figsize = (figsize, figsize)
    else:
        raise ValueError(
            f"Unsupported type of `figsize`, got `{type(figsize)}`. "
        )

    # get `num_plot`, `max_len_y_list`
    if isinstance(y_list_list, np.ndarray):
        num_plot = y_list_list.shape[0]
        max_len_y_list = y_list_list.shape[1]
    elif isinstance(y_list_list, list):
        num_plot = len(y_list_list)
        
        # y_list_list: List[List[float]]
        if isinstance(y_list_list[0], list):
            max_len_y_list = max(
                [len(y_list) for y_list in y_list_list]
            )
        # y_list_list: List[np.ndarray]
        else:
            max_len_y_list = max(
                [y_list.shape[0] for y_list in y_list_list]
            )
    else:
        raise ValueError(
            f"Unsupported type of `y_list_list`, got `{type(y_list_list)}`. "
        )

    # get `x_list`
    if x_list is None:
        x_list = [i for i in range(max_len_y_list)]
    
    # get `y_label_list`
    y_label_list = tsfm_to_1d_array(
        array = y_label_list, 
        target_length = num_plot, 

        dtype = type("str")
    )

    if marker_list is None:
        marker_list = ['o']
    
    if marker_size_list is None:
        marker_size_list = [6]
    
    if color_list is not None:
        color_list = tsfm_to_1d_array(
            array = color_list, 
            target_length = num_plot, 

            dtype = type("str")
        )

    if alpha_list is not None:
        alpha_list = tsfm_to_1d_array(
            array = alpha_list, 
            target_length = num_plot, 

            dtype = type("str")
        )

    if fill_alpha_list is not None:
        fill_alpha_list = tsfm_to_1d_array(
            array = fill_alpha_list, 
            target_length = num_plot, 

            dtype = type("str")
        )

    if face_color_list is not None:
        face_color_list = tsfm_to_1d_array(
            array = face_color_list, 
            target_length = num_plot, 

            dtype = type("str")
        )

    fig, ax = plt.subplots(figsize = figsize)

    # plot
    for plot_idx, y_list in enumerate(y_list_list):
        ax.plot(
            x_list, y_list, 
            marker = marker_list[plot_idx % len(marker_list)], 
            markersize = marker_size_list[plot_idx % len(marker_size_list)], 
            label = y_label_list[plot_idx], 
            color = None if (color_list is None) else color_list[plot_idx], 
            alpha = None if (alpha_list is None) else alpha_list[plot_idx]
        )

    # fill between
    if confidence_level is not None:
        y_list_list = np.asarray(y_list_list)

        std_list_list = np.asarray(std_list_list)

        # TODO: update
        y_down_list_list = y_list_list - 1.96 * std_list_list / np.sqrt(num_sample)
        y_up_list_list = y_list_list + 1.96 * std_list_list / np.sqrt(num_sample)

        # plot
        for (
            plot_idx, 
            (y_down_list, y_up_list)
        ) in enumerate(
            zip(y_down_list_list, y_up_list_list)
        ):
            ax.fill_between(
                x_list, 
                y_up_list, y_down_list, 

                alpha = fill_alpha_list[plot_idx], 
                facecolor = face_color_list[plot_idx]
            )

    # set title
    if plot_title is not None:
        ax.set_title(plot_title)
    
    # set x-label
    if plot_x_label is not None:
        ax.set_xlabel(plot_x_label)

    # set y-label
    if plot_y_label is not None:
        ax.set_ylabel(plot_y_label)

    # determine whether to show grid
    ax.grid(show_grid)

    # determine whether to show legend
    if show_legend:
        ax.legend()

    fig.tight_layout()

    # `get_line_chart()` done
    return fig, ax

# TODO




def get_scatter(
    figsize: Optional[Union[float, Tuple[float, float]]] = (10, 5), 

    point_list: Optional[List[Tuple[float]]] = None, 

    label_list: Optional[List[str]] = None, 

    marker_list: Union[str, List[str]] = [
        'o',  # circle
        's',  # square
        '*',  # star
        '+',  # plus
        'x',  # x

        'd',  # diamond
        'D',  # thin diamond

        '^',  # up-pointing triangle
        'v',  # down-pointing triangle
        '<',  # left-pointing triangle
        '>',  # right-pointing triangle

        'p',  # pentagon
        'h',  # hexagon 1
        'H',  # hexagon 2
    ], 

    color_list: Optional[List[str]] = None, 

    area_list: Optional[List[float]] = None, 

    plot_title: Optional[str] = None, 
    plot_x_label: Optional[str] = None, 
    plot_y_label: Optional[str] = None, 

    show_grid: Optional[bool] = True, 

    # legend
    show_legend: Optional[bool] = True, 
    legend_num_col: Optional[int] = 1
) -> Tuple: 
    # get `figsize`
    if isinstance(figsize, tuple):
        pass
    elif isinstance(figsize, float):
        figsize = (figsize, figsize)
    else:
        raise ValueError(
            f"Unsupported type of `figsize`, got `{type(figsize)}`. "
        )
    
    fig, ax = plt.subplots(figsize = figsize)

    for i, point in enumerate(point_list):
        ax.scatter(
            x = point[0], y = point[1], 

            label = label_list[i], 

            marker = None if (marker_list is None) else marker_list[i], 

            c = None if (color_list is None) else color_list[i], 

            s = None if (area_list is None) else area_list[i], 
        )

    # show labels
    if plot_x_label is not None:
        ax.set_xlabel(plot_x_label)
    if plot_y_label is not None:
        ax.set_ylabel(plot_y_label)

    # show title
    if plot_title is not None:
        ax.set_title(plot_title)
    
    # determine whether to show grid
    ax.grid(show_grid)

    # determine whether to show legend
    if show_legend:
        ax.legend(ncol = legend_num_col)

    fig.tight_layout()

    return fig, ax
