# tmux
tmux                         # 新建会话
tmux new -s dev              # 新建名为 dev 的会话
tmux ls                      # 查看所有会话
tmux attach -t dev           # 进入 dev 会话
tmux attach -d -t dev        # 踢掉其他客户端并进入 dev
tmux new -A -s dev           # 会话存在则进入，不存在则创建
tmux rename-session -t dev work
tmux kill-session -t dev     # 删除指定会话
tmux kill-server             # 关闭所有 tmux 会话

# 训练
conda activate pi_env
export WANDB_MODE=offline
export WANDB_SILENT=true
cd ~/minipi/mini_pi_walk
python -m humanoid.scripts.train \
  --task=pai_ppo \
  --experiment_name=baseline \
  --run_name=v1 \
  --headless \
  --num_envs=2048 \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --graphics_device_id=-1 \
  --max_iterations=100

续训
python scripts/train.py \
  --task pai_ppo \
  --experiment_name Pai_reward \
  --resume \
  --load_run Oct08_14-43-36_net3 \
  --checkpoint 10001 \
  --max_iterations 2000 \
  --num_envs 2048 \
  --run_name track_ang_vel_30_3 \
  --headless
# 评测最新model

python -m humanoid.scripts.play.py \
  --task pai_ppo \
  --experiment_name Pai_reward \
  --load_run Oct27_14-27-01_feet_dist_1\
  --checkpoint -1 \
  --num_envs 64 \
  --headless \
  --rl_device=cuda:1 \
  --sim_device=cuda:1

# sim2sim测评
python scripts/sim2sim.py \
  --headless \
  --load_model /home/jingwei/minipi/Mini_Pi_RL_Baseline/logs/Pai_stage0/exported/policies/policy_torch.pt

# 模型格式转换
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/nvidia/{path to your onnx file}/{your policy name}.onnx \
  --saveEngine=/home/{your save path}/{your policy name}.trt

# 真机部署
pkill -f ros
cd sim2real_master
source ./install/setup.bash
roslaunch sim2real_master joy_control_pi.launch
# 
conda activate pi_env
cd ~/minipi/Mini_Pi_RL_Baseline
python -m pip install 'protobuf==4.25.9'
python -m pip check
# tensorboard
lsof -i :6006
kill
tensorboard --logdir logs/Pai_stage0 --host 127.0.0.1 --port 6006
tensorboard --logdir /home/jingwei/minipi/Mini_Pi_RL_Baseline/logs/Pai_stage0/Sep24_13-04-50_no_any_disturbance --port=6008
# 复制
scp -P 5646 jingwei:/home/jingwei/minipi/result.zip E:\复现\Mini_Pi_Pro
# 压缩
zip -r result.zip result
