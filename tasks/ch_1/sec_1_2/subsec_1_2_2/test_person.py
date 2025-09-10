# Note: 
# 本程序需在 `tasks/ch_1/sec_1_2/subsec_1_2_2` 目录下运行

from my_class.person import (
    get_person_1, 
    get_person_2, 
    Person
)


# 若需用 `Person`, 则需 import
def display_person_1(
    person: Person
):
    print(f"name: {person.name}")
    print(f"age: {person.age}")

    # `display_person_1()` done
    pass


# 未实际用 `Person` 类 (如创建对象), 只是标注类型
def display_person_2(
    person: "Person"
):
    print(f"name: {person.name}")
    print(f"age: {person.age}")

    # `display_person_1()` done
    pass


person = get_person_1(
    name = "Hytidel", 
    age = 114514
)

display_person_1(
    person = person
)
