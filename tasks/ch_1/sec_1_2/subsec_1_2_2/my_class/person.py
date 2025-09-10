# 函数在 `Person` 类前定义
def get_person_1(
    name: str, 
    age: int
) -> "Person":
# ) -> Person:  # 报错, `Person` 未定义
    if age < 0:
        raise ValueError(
            f"Illegal `age`, got `{age}`. "
        )

    person = Person(
        name = name, 
        age = age
    )

    # `get_person_1()` done
    return person


class Person:
    def __init__(
        self, 

        name: str, 
        age: int
    ):
        self.name = name
        self.age = age

        # `__init__()` done
        pass


# 函数在 `Person` 类后定义
def get_person_2(
    name: str, 
    age: int
) -> Person:
    if age < 0:
        raise ValueError(
            f"Illegal `age`, got `{age}`. "
        )

    person = Person(
        name = name, 
        age = age
    )

    # `get_person_2()` done
    return person
