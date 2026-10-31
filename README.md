# mini_pi_walk

基于 NVIDIA Isaac Gym 的 Pi 双足机器人强化学习项目，支持 PPO 训练及 MuJoCo sim2sim 验证。
## 环境准备

### 1. 系统与 NVIDIA 驱动

以下命令在 Linux（Ubuntu）终端中执行。

准备 NVIDIA GPU、可用的显卡驱动，以及 Miniconda 或 Anaconda。原基线要求驱动版本至少为 515，并给出以下 Ubuntu 安装示例；具体驱动版本还需满足所选 PyTorch CUDA 运行时的要求：

```bash
# 仅在需要安装驱动时执行，按系统和显卡选择合适的驱动版本
sudo apt update
sudo apt install nvidia-driver-525
```

安装驱动后按系统提示重启，再检查 GPU 是否可见：

```bash
nvidia-smi
```

原文展示的是驱动 550、CUDA 12.4 的环境。不要将驱动 525 的安装示例直接视为后续 CUDA 12.4 示例的配套配置；应先确认实际驱动支持所选运行时。`nvidia-smi` 中的 CUDA Version 表示驱动支持的 CUDA 版本，并不代表已安装同版本 CUDA Toolkit。

### 2. 创建 Python 环境


```bash
conda create -n pi_env python=3.8 -y
conda activate pi_env
python --version
```

每次打开新的终端，训练或仿真前都需要执行 `conda activate pi_env`。

### 3. 安装 PyTorch 与 NumPy

原文使用 CUDA 12.4。下面给出固定 PyTorch 版本的安装示例，避免不指定版本时随软件更新改变依赖组合；此组合仍需在目标 GPU 上通过后面的环境检查。安装命令参见 [PyTorch 历史版本官方说明](https://pytorch.org/get-started/previous-versions/#v241)。

```bash
# 适用于驱动支持 CUDA 12.4 的环境
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# 与本项目 setup.py 的精确版本要求一致
conda install numpy=1.23.5
```

若驱动不支持 CUDA 12.4，请从上述官方说明选择与驱动、GPU 和 Python 3.8 相容的组合。不要直接照搬“安装最新版 PyTorch”，也不要仅修改 CUDA 版本而忽略其他包的兼容关系。

检查 PyTorch 是否能使用 GPU：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

用于 GPU 训练时，`CUDA available` 应为 `True`。

### 4. 安装 Isaac Gym Preview 4

从 [NVIDIA Isaac Gym 官方页面](https://developer.nvidia.com/isaac-gym) 的归档入口下载 **Isaac Gym Preview 4** 并解压。本项目使用该版本的 `isaacgym` 接口。

将 `/path/to/isaacgym` 替换为实际解压路径：

```bash
conda activate pi_env
cd /path/to/isaacgym/python
python -m pip install -e .

# 在有图形显示的会话中运行官方示例，验证仿真和渲染
cd examples
python 1080_balls_of_solitude.py
```

示例应能启动仿真窗口。若运行失败，先参阅解压目录中的 `isaacgym/docs/index.html`，确认驱动、动态库和显示环境；无图形显示的服务器可在后续项目检查中使用 `--headless`。

### 5. 安装本项目依赖

进入本项目包含 `setup.py` 的根目录，将占位路径替换为 Linux 主机上的实际路径：

```bash
conda activate pi_env
cd /path/to/mini_pi_walk
python -m pip install -e .

# sim2sim.py 直接使用 SciPy，但本项目 setup.py 未显式声明它
python -m pip install scipy
python -m pip check
```

`pip install -e .` 会按 `setup.py` 安装项目及其依赖，包括 `wandb`、`tensorboard`、`tqdm`、`numpy==1.23.5`、`opencv-python`、`mujoco==2.3.6`、`mujoco-python-viewer` 和 `matplotlib`。Isaac Gym 应先按上一步从下载的源码目录安装。


### 6. 环境检查与最小训练验证

在 `mini_pi_walk` 根目录执行：

```bash
# Isaac Gym 应在 PyTorch 前导入；输出路径应指向 mini_pi_walk/humanoid
python -c "import isaacgym; import torch, numpy, mujoco, scipy, humanoid; print('humanoid:', humanoid.__file__); print('NumPy:', numpy.__version__); print('MuJoCo:', mujoco.__version__); print('CUDA available:', torch.cuda.is_available())"

# 使用少量并行环境和一次训练迭代检查完整流程；会生成训练日志
python humanoid/scripts/train.py --task=pai_ppo --run_name env_check --headless --num_envs 64 --max_iterations 1
```

依赖检查通过且最小训练能够完成后，再按显存容量增加 `--num_envs`；原基线的训练示例使用 4096 个环境。当前仓库的脚本位于 `humanoid/scripts/`，运行时应使用这里的路径。

## 常用命令
以下为无头服务器用命令
### 1. 训练
```bash
conda activate pi_env
export WANDB_MODE=offline
export WANDB_SILENT=true
cd /mini_pi_walk
python -m humanoid.scripts.train \
  --task=pai_ppo \
  --experiment_name=baseline \
  --run_name=v1 \
  --headless \
  --num_envs=2048 \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --graphics_device_id=-1 \
  --max_iterations=10001
```

若要续训
```bash
conda activate pi_env
export WANDB_MODE=offline
export WANDB_SILENT=true
cd /mini_pi_walk
python -m humanoid.scripts.train \
  --task pai_ppo \
  --experiment_name baseline \
  --resume \
  --load_run Oct31_00-29-08_v1 \
  --checkpoint 10001 \
  --max_iterations 2000 \
  --num_envs 2048 \
  --run_name v1 \
  --headless
```

### 2. 评测与导出
```bash
conda activate pi_env
export WANDB_MODE=offline
export WANDB_SILENT=true
cd /mini_pi_walk
python -m humanoid.scripts.play \
  --task pai_ppo \
  --experiment_name baseline \
  --load_run Oct31_00-29-08_v1 \
  --checkpoint -1 \
  --num_envs 64 \
  --headless \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

### 3. sim2sim测评
```bash
conda activate pi_env
export WANDB_MODE=offline
export WANDB_SILENT=true
cd /mini_pi_walk
python -m humanoid.scripts.sim2sim \
  --headless \
  --load_model logs/baseline/exported/policies/policy_torch.pt
```

## 真机部署
假设导出模型test.onnx
1. 将onnx格式policy放在
```bash
/home/nvidia/sim2real_master-devel-bydmmc_and_amp/install/share/sim2real/policy/walk
```
2. 模型格式转换
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/nvidia/sim2real_master-devel-bydmmc_and_amp/install/share/sim2real/policy/walk/test.onnx \
  --saveEngine=/home/nvidia/sim2real_master-devel-bydmmc_and_amp/install/share/sim2real/policy/walk/test.trt
```
3. 配置参数重启
打开
```bash
/home/nvidia/sim2real_master-devel-bydmmc_and_amp/install/share/sim2real/config/walk/humanoidgym.yaml
```
将policy_name改为
```bash
policy_name: "walk/test.trt"
```

4. 重启ros
```bash
pkill -f ros
cd sim2real_master
source ./install/setup.bash
roslaunch sim2real_master joy_control_pi.launch
```