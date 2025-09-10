import configparser
import configparser as cp
from concurrent.futures import process


print(configparser.__name__)
print(cp.__name__)  # 仍为 `configparser`
print(process.__name__)

print(__name__)
