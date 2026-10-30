import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

def quat_apply_yaw(quat,vec):
    # quat：姿态四元数，分量顺序为 [x, y, z, w]；vec：要旋转的三维向量。
    # 目标是构造只绕 z 轴旋转的四元数，再用它旋转 vec。
    # clone() 复制数据，避免下面的原地赋值修改传入的 quat。
    # view(-1, 4) 将四元数整理为 N 行、4 列；-1 表示自动计算行数。
    quat_yaw=quat.clone().view(-1,4)
    # 对所有行，将 x、y 分量置零，保留 z、w 分量。
    # 四元数的 x、y、z 并不是 roll、pitch、yaw 三个欧拉角；
    # 这种投影一般不等同于从原姿态严格提取欧拉 yaw。
    quat_yaw[:,:2]=0
    # 置零后长度可能不再为 1，逐行归一化以构造单位四元数。
    # 前提是保留的 z、w 不同时为零，否则无法得到有效的单位四元数。
    quat_yaw=normalize(quat_yaw)
    # 将得到的绕 z 轴旋转应用到 vec：改变水平朝向，保持向量的 z 分量。
    # vec 中的三维向量数量应与四元数数量匹配；返回旋转后的向量。
    return quat_apply(quat_yaw,vec)

def wrap_to_pi(angles):
    # 先对2pi取余
    angles%=2*np.pi
    # 区间平移至[-pi,pi]
    angles-=2*np.pi*(angles>np.pi)
    return angles

def torch_rand_sqrt_float(lower,upper,shape,device):
    # 在 [lower, upper] 范围内生成随机数，分布偏向两端，中间附近较少。
    # shape 指定输出张量形状，device 指定生成位置，例如 "cpu" 或 "cuda:0"。
    # torch.rand 先生成 [0, 1) 上的均匀随机数，再映射到 [-1, 1)。
    # *shape 将形状展开为各维度参数，例如 (2, 3) 展开为 2, 3。
    r=2*torch.rand(*shape,device=device)-1
    # 保留正负号，对绝对值开平方：负数取 -sqrt(-r)，非负数取 sqrt(r)。
    # 例如 -0.25 -> -0.5，0.25 -> 0.5，使随机数远离 0、偏向 -1 和 1。
    # torch.where 按元素选择结果；原来的 -t 是笔误，此处应为 -r。
    r=torch.where(r<0.,-torch.sqrt(-r),torch.sqrt(r))
    # 将 [-1, 1] 映射到 [0, 1]；分布偏向 0 和 1，而非均匀分布。
    r=(r+1.)/2.
    # 缩放到目标区间 [lower, upper]，保留偏向区间两端的分布特征。
    return (upper-lower)*r+lower
