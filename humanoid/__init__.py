import os
# 获取当前文件的绝对真实路径，再向上取两级目录，作为项目根目录
# __file__：当前 Python 文件的路径
# os.path.realpath()：转换为绝对路径，并解析符号链接
# os.path.dirname()：获取上一级目录，嵌套两次就是向上取两级
LEGGED_GYM_ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)

# 将项目根目录与 'humanoid'、'envs' 拼接，得到环境代码目录的路径
LEGGED_GYM_ENVS_DIR = os.path.join(
    LEGGED_GYM_ROOT_DIR, 'humanoid', 'envs'
)