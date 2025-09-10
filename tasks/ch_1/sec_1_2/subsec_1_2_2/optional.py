from typing import Optional


# Python 3.10+
def process_num_list_1(
    num_list: list[float], 

    display_output: bool | None = False
):
    num_list = [
        int(num) \
            for num in num_list
    ]

    if display_output:
        print(f"num_list: {num_list}")

    # `process_num_list_1()` done
    pass


# Python 3.9 及更早版本
def process_num_list_2(
    num_list: list[float], 

    display_output: Optional[bool] = False
) -> None:  # 无返回值也可显式标注
    num_list = [
        int(num) \
            for num in num_list
    ]

    if display_output:
        print(f"num_list: {num_list}")

    # `process_num_list_1()` done
    pass
