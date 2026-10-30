import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import PaiCfg
import torch
import threading
import glfw
import matplotlib.pyplot as plt
import time  # 新增导入time模块

class cmd:
    vx = 0.
    vy = 0.
    dyaw = 0.

class mujoco_visual:
    def __init__(self, headless=False) -> None:
        self.headless = headless
        self.count_lowlevel = 0
        self.stop_event = threading.Event()
        self.vel = [0,0,0]
        self.w = [0,0,0]
        # 键位映射（避开图片中的C/J/G/M/T/V/E）
        self.key_states = {
            # 方向控制
            glfw.KEY_UP: False,
            glfw.KEY_DOWN: False,
            glfw.KEY_LEFT: False,
            glfw.KEY_RIGHT: False,
            glfw.KEY_W: False,  # 新增
            glfw.KEY_S: False,   # 新增
            glfw.KEY_A: False,   # 新增
            glfw.KEY_D: False,   # 新增
            # 图片功能键（保持原有）
            glfw.KEY_C: False,
            glfw.KEY_J: False,
            glfw.KEY_G: False,
            glfw.KEY_M: False,
            glfw.KEY_T: False,
            glfw.KEY_V: False,
            glfw.KEY_E: False
        }

    def quaternion_to_euler_array(self, quat):
        x, y, z, w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        
        return np.array([roll_x, pitch_y, yaw_z])

    def get_obs(self, data):
        q = data.qpos.astype(np.double)
        dq = data.qvel.astype(np.double)
        quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        r = R.from_quat(quat)
        v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
        self.vel = [v[0], v[1], v[2]]
        omega = data.sensor('angular-velocity').data.astype(np.double)
        self.w = omega
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        return (q, dq, quat, v, omega, gvec)

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        return (target_q - q) * kp + (target_dq - dq) * kd

    def plot_thread(self):
        plt.ion()
        fig, ax = plt.subplots()
        t, cmd_x, true_x = [], [], []
        line_cmd_x, = ax.plot(t, cmd_x, 'r-')
        line_true_x, = ax.plot(t, true_x, 'b-')

        while not self.stop_event.is_set():
            t.append(self.count_lowlevel * 0.001)
            cmd_x.append(cmd.vx)
            true_x.append(self.vel[0])
            line_cmd_x.set_xdata(t)
            line_cmd_x.set_ydata(cmd_x)
            line_true_x.set_xdata(t)
            line_true_x.set_ydata(true_x)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)
        plt.savefig('sine_wave.png')
        plt.ioff()
        plt.close('all')

    def run_mujoco(self, policy, cfg):
        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
        model.opt.timestep = cfg.sim_config.dt
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        viewer = None if self.headless else mujoco_viewer.MujocoViewer(model, data)

        def key_callback(window, key, scancode, action, mods):
            # 处理运动控制键
            if key in self.key_states:
                self.key_states[key] = action == glfw.PRESS
            
            # 实现图片中的可视化功能（保持原有）
            if key == glfw.KEY_C and action == glfw.PRESS:
                viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] ^= 1
            if key == glfw.KEY_J and action == glfw.PRESS:
                viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] ^= 1
            if key == glfw.KEY_G and action == glfw.PRESS:
                viewer._graph._visible ^= 1
            if key == glfw.KEY_M and action == glfw.PRESS:
                viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_COM] ^= 1

        if not self.headless:
            glfw.set_key_callback(viewer.window, key_callback)
            self.window = viewer.window

        target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
        action = np.zeros((cfg.env.num_actions), dtype=np.double)

        hist_obs = deque()
        for _ in range(cfg.env.frame_stack):
            hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

        for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
            if not self.headless and glfw.window_should_close(self.window):
                break

            if not self.headless:
                # 运动控制处理（新增WSAD）
                cmd.vx = 0.5 * (self.key_states[glfw.KEY_UP] or self.key_states[glfw.KEY_W])
                cmd.vx -= 0.5 * (self.key_states[glfw.KEY_DOWN] or self.key_states[glfw.KEY_S])
                cmd.vy = 0.5 * (self.key_states[glfw.KEY_A] - self.key_states[glfw.KEY_D])
                cmd.dyaw = 1.0 * (self.key_states[glfw.KEY_LEFT] - self.key_states[glfw.KEY_RIGHT])

            q, dq, quat, v, omega, gvec = self.get_obs(data)
            q = q[-cfg.env.num_actions:]
            dq = dq[-cfg.env.num_actions:]
            for i in range(6):
                q[i], q[i+6] = q[i+6], q[i]
                dq[i], dq[i+6] = dq[i+6], dq[i]

            if self.count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = self.quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                # 与 Isaac Gym 训练环境的 rewards.cycle_time = 0.4 保持一致。
                obs[0, 0] = math.sin(2 * math.pi * self.count_lowlevel * cfg.sim_config.dt / 0.4)
                obs[0, 1] = math.cos(2 * math.pi * self.count_lowlevel * cfg.sim_config.dt / 0.4)
                obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
                obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
                obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 29:41] = action
                obs[0, 41:44] = omega
                obs[0, 44:47] = eu_ang

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                target_q = action * cfg.control.action_scale

            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            tau = self.pd_control(target_q, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            for i in range(6):
                tau[i], tau[i+6] = tau[i+6], tau[i]
            data.ctrl = tau

            mujoco.mj_step(model, data)

            if not self.headless:
                time.sleep(0.001)#步伐频率加快建议设置0.001，步伐频率加快建议设置0.01
                if self.count_lowlevel % cfg.sim_config.decimation == 0:
                    viewer.render()
            self.count_lowlevel += 1

        self.stop_event.set()
        if viewer is not None:
            viewer.close()

if __name__ == '__main__':
    import argparse
    print(LEGGED_GYM_ROOT_DIR)
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=False,
                        help='Run to load from.',
                        default=f"{LEGGED_GYM_ROOT_DIR}/logs/Pai_ppo/exported/policies/policy_1.pt")
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    parser.add_argument('--headless', action='store_true', help='Run without GLFW viewer')
    args = parser.parse_args()

    class Sim2simCfg(PaiCfg):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = [40,20,20,40,40,20]*(2)
            kds = [0.6,0.4,0.4,0.6,0.6,0.4]*(2)
            # URDF effort=21 Nm，训练环境按 safety.torque_limit=0.85 缩放。
            tau_limit = 17.85 * np.ones(12, dtype=np.double)

    policy = torch.jit.load(args.load_model)
    visualizer = mujoco_visual(headless=args.headless)

    if args.headless:
        visualizer.run_mujoco(policy, Sim2simCfg())
        print("Headless simulation completed.")
    else:
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        try:
            matplotlib_thread = threading.Thread(target=visualizer.plot_thread)
            mujoco_thread = threading.Thread(target=visualizer.run_mujoco, args=(policy, Sim2simCfg()))
            matplotlib_thread.start()
            mujoco_thread.start()
            matplotlib_thread.join()
            mujoco_thread.join()
        finally:
            glfw.terminate()
        print("Simulation completed. GLFW terminated.")
