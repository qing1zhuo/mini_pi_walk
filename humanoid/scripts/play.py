import csv
import os
from datetime import datetime
from statistics import mean

from isaacgym import gymapi
from isaacgym.torch_utils import *

import cv2
import numpy as np
import torch
from tqdm import tqdm

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import (export_policy_as_jit, export_policy_to_onnx,
                            get_args, get_load_path, set_seed, task_registry)
from humanoid.utils.helpers import update_cfg_from_args
from humanoid.utils.math import wrap_to_pi

# x/y are body-frame velocity commands; heading is a world-frame target angle.
COMMANDS = (
    ("stand", 0.0, 0.0, 0.0),
    ("forward_slow", 0.3, 0.0, 0.0),
    ("forward_fast", 0.6, 0.0, 0.0),
    ("backward", -0.3, 0.0, 0.0),
    ("left", 0.0, 0.3, 0.0),
    ("right", 0.0, -0.3, 0.0),
    ("turn_left", 0.0, 0.0, np.pi / 2),
    ("turn_right", 0.0, 0.0, -np.pi / 2),
    ("turn_around", 0.0, 0.0, np.pi),
    ("forward_turn_left", 0.3, 0.0, np.pi / 2),
    ("forward_turn_right", 0.3, 0.0, -np.pi / 2),
)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 64)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.noise.add_noise = True
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s + 1.0
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False  # apply scheduled pushes ourselves
    seed = args.seed if args.seed is not None else 123145
    env_cfg.seed = train_cfg.seed = seed
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    train_cfg.runner.resume = True
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    checkpoint = get_load_path(log_root, train_cfg.runner.load_run, train_cfg.runner.checkpoint)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if not args.headless:
        env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(log_root, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        export_policy_to_onnx(ppo_runner.alg.actor_critic, path)
        print("Exported policy to:", path)

    output_dir = os.path.join(os.path.dirname(checkpoint), "eval",
                              datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{seed}")
    os.makedirs(output_dir, exist_ok=True)
    video_dir = os.path.join(output_dir, "videos")
    if RENDER:
        os.makedirs(video_dir, exist_ok=True)
    env.record_eval_rewards = True
    friction = env.friction_coeffs.view(-1).tolist()
    base_mass = [env.gym.get_actor_rigid_body_properties(env.envs[i],
                 env.gym.get_actor_handle(env.envs[i], 0))[0].mass for i in range(env.num_envs)]
    reward_names = list(env.episode_sums)
    fields = ["case", "env_id", "command_vx", "command_vy", "target_heading", "friction", "base_mass",
              "steps", "seconds", "timeout", "total_reward", "vx_mae", "vy_mae", "yaw_rate_mae", "heading_mae"]
    fields += ["reward_" + name for name in reward_names]
    push_interval = int(round(env.cfg.domain_rand.push_interval_s / env.dt))
    all_rows = []

    camera = None
    if RENDER:
        camera = env.camera_handle  # BaseTask creates this sensor, including in headless mode.
        offset = gymapi.Vec3(1, -1, 0.5)
        rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
        actor = env.gym.get_actor_handle(env.envs[0], 0)
        body = env.gym.get_actor_rigid_body_handle(env.envs[0], actor, 0)
        env.gym.attach_camera_to_body(camera, env.envs[0], body,
                                      gymapi.Transform(offset, rotation), gymapi.FOLLOW_POSITION)

    with open(os.path.join(output_dir, "episodes.csv"), "w", newline="", encoding="utf-8") as episodes_file, \
         open(os.path.join(output_dir, "pushes.csv"), "w", newline="", encoding="utf-8") as pushes_file:
        episodes = csv.DictWriter(episodes_file, fieldnames=fields)
        episodes.writeheader()
        pushes = csv.writer(pushes_file)
        pushes.writerow(["case", "step", "env_id", "active", "push_vx", "push_vy", "push_wx", "push_wy", "push_wz"])

        for name, vx, vy, heading in COMMANDS:
            set_seed(seed + 1)  # same initial joint/phase/noise samples in every case
            env.reset_idx(torch.arange(env.num_envs, device=env.device))
            env.last_root_vel.zero_()
            env.last_contacts.zero_()
            env.feet_height.zero_()
            env.last_feet_z = 0.05
            env.rand_push_force.zero_()
            env.rand_push_torque.zero_()
            env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
            env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
            env.commands[:, 0] = vx
            env.commands[:, 1] = vy
            env.commands[:, 3] = heading
            env.commands[:, 2] = torch.clamp(0.5 * wrap_to_pi(heading - env.base_euler_xyz[:, 2]), -1.0, 1.0)
            env.compute_observations()  # policy sees this case's command before its first action
            obs = env.get_observations()

            active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
            steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            timeouts = torch.zeros_like(active)
            reward = torch.zeros(env.num_envs, device=env.device)
            errors = torch.zeros(env.num_envs, 4, device=env.device)
            terms = {}
            video = None
            if RENDER:
                video_path = os.path.join(video_dir, name + ".mp4")
                video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                        1.0 / env.dt, (720, 480))
                if not video.isOpened():
                    raise RuntimeError("Cannot open video output: " + video_path)
            try:
                for step in tqdm(range(int(env.max_episode_length) + 2), desc=name):
                    if step % push_interval == 0:
                        with torch.random.fork_rng(devices=[env.sim_device_id] if env.device != "cpu" else []):
                            torch.manual_seed(seed + 100000 + step // push_interval)
                            env._push_robots()
                        samples = torch.cat((env.rand_push_force[:, :2], env.rand_push_torque), dim=1).cpu().tolist()
                        active_ids = active.cpu().tolist()
                        pushes.writerows([name, step, i, int(active_ids[i]), *sample]
                                         for i, sample in enumerate(samples))

                    command = env.commands[:, :3].clone()
                    heading_error = torch.abs(wrap_to_pi(heading - env.base_euler_xyz[:, 2]))
                    with torch.no_grad():
                        actions = policy(obs.detach())
                    obs, _, rews, dones, infos = env.step(actions.detach())
                    steps[active] += 1
                    reward[active] += rews[active]
                    errors[active, 0] += torch.abs(env.base_lin_vel[active, 0] - command[active, 0])
                    errors[active, 1] += torch.abs(env.base_lin_vel[active, 1] - command[active, 1])
                    errors[active, 2] += torch.abs(env.base_ang_vel[active, 2] - command[active, 2])
                    errors[active, 3] += heading_error[active]

                    finished = active & dones.bool()
                    if torch.any(finished):
                        timeouts[finished] = infos["time_outs"][finished]
                        finished_ids = set(finished.nonzero(as_tuple=True)[0].tolist())
                        for position, env_id in enumerate(env.eval_episode_ids.tolist()):
                            if env_id in finished_ids:
                                terms[env_id] = {key: float(values[position])
                                                 for key, values in env.eval_episode_rewards.items()}
                    if video is not None and active[0] and not dones[0]:
                        env.gym.fetch_results(env.sim, True)
                        env.gym.step_graphics(env.sim)
                        env.gym.render_all_camera_sensors(env.sim)
                        image = env.gym.get_camera_image(env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR)
                        video.write(cv2.cvtColor(np.reshape(image, (480, 720, 4)), cv2.COLOR_RGBA2BGR))
                    active &= ~dones.bool()
                    if not torch.any(active):
                        break
            finally:
                if video is not None:
                    video.release()
            if torch.any(active):
                raise RuntimeError(f"{name}: some environments did not finish within {int(env.max_episode_length) + 2} steps")

            counts = steps.cpu().tolist()
            totals = reward.cpu().tolist()
            error_values = errors.cpu().tolist()
            timeout_values = timeouts.cpu().tolist()
            for i in range(env.num_envs):
                row = {"case": name, "env_id": i, "command_vx": vx, "command_vy": vy,
                       "target_heading": heading, "friction": friction[i], "base_mass": base_mass[i],
                       "steps": counts[i], "seconds": counts[i] * env.dt,
                       "timeout": int(timeout_values[i]), "total_reward": totals[i],
                       "vx_mae": error_values[i][0] / counts[i], "vy_mae": error_values[i][1] / counts[i],
                       "yaw_rate_mae": error_values[i][2] / counts[i],
                       "heading_mae": error_values[i][3] / counts[i]}
                row.update({"reward_" + key: value for key, value in terms.get(i, {}).items()})
                episodes.writerow(row)
                all_rows.append(row)
            episodes_file.flush()
            pushes_file.flush()

    with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as report:
        report.write(f"Checkpoint: {checkpoint}\nSeed: {seed}\nTask: {args.task}\nEnvironments: {env.num_envs}\n")
        report.write(f"Episode limit: {env.cfg.env.episode_length_s}s; policy dt: {env.dt}s\n")
        report.write(f"Friction range: {env.cfg.domain_rand.friction_range}; fixed per env, see episodes.csv\n")
        report.write(f"Base mass range: {env.cfg.domain_rand.added_mass_range}; fixed per env, see episodes.csv\n")
        report.write(f"Push schedule: step 0, then every {env.cfg.domain_rand.push_interval_s}s; velocity overwrite bounds: ")
        report.write(f"xy={env.cfg.domain_rand.max_push_vel_xy}, angular={env.cfg.domain_rand.max_push_ang_vel}\n")
        report.write("Push samples and active status: pushes.csv; same samples at each scheduled step for every case.\n")
        report.write("Other stochastic effects: observation noise and action delay/noise; reset seed is identical for every case.\n")
        report.write("Reward columns are weighted episode sums before total-reward clipping; they need not sum to total_reward.\n")
        report.write("Timeout=1 means the episode reached its time limit; timeout=0 means early termination.\n")
        report.write("GPU physics may not reproduce bit for bit across runs.\n")
        report.write(f"Videos: {video_dir if RENDER else 'disabled'}\n\n")
        for name, vx, vy, heading in COMMANDS:
            rows = [row for row in all_rows if row["case"] == name]
            report.write(f"{name}: vx={vx}, vy={vy}, target_heading={heading:.3f}; ")
            report.write(f"timeout_rate={mean(row['timeout'] for row in rows):.3f}, ")
            report.write(f"mean_reward={mean(row['total_reward'] for row in rows):.3f}, ")
            report.write(f"vx_mae={mean(row['vx_mae'] for row in rows):.3f}, ")
            report.write(f"vy_mae={mean(row['vy_mae'] for row in rows):.3f}, ")
            report.write(f"yaw_rate_mae={mean(row['yaw_rate_mae'] for row in rows):.3f}, ")
            report.write(f"heading_mae={mean(row['heading_mae'] for row in rows):.3f}\n")
    print("Evaluation saved to:", output_dir)


if __name__ == "__main__":
    EXPORT_POLICY = True
    RENDER = True
    args = get_args()
    play(args)
