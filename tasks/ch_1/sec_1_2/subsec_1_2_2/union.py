from typing import Union


# Python 3.10+
def num_to_str_1(
    num: int | float
) -> str:
    # `num_to_str_1()` done
    return str(num)


# Python 3.9 及更早版本
def num_to_str_2(
    num: Union[int, float]
) -> str:
    # `num_to_str_2()` done
    return str(num)


# 嵌套
def num_list_to_str_list(
    num_list: list[int | float]
) -> list[str]:
    str_list = [
        str(num) \
            for num in num_list
    ]

    # `num_list_to_str_list()` done
    return str_list
