from typing import List, Set, Dict, Tuple


# 列表
x: List[float] = [1.0]

# 集合
x: Set[int] = {6, 7}

# 字典
x: Dict[str, float] = {
    "field": 2.0
}

# 元组
x: Tuple[int, str, float] = (3, "yes", 7.5)

# 变长元组
x: Tuple[int, ...] = (1, 2, 3)
