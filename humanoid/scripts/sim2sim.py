import argparse
import csv
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from humanoid import LEGGED_GYM_ROOT_DIR


# x/y are body-frame velocity commands; heading is a world-frame target angle.
# Keep this list aligned with humanoid/scripts/play.py.
COMMANDS = (
    ("stand", 0.0, 0.0, 0.0),
    ("forward_slow", 0.3, 0.0, 0.0),
    ("forward_fast", 0.6, 0.0, 0.0),
    ("backward", -0.3, 0.0, 0.0),
    ("left", 0.0, 0.3, 0.0),
    ("right", 0.0, -0.3, 0.0),
    ("turn_left", 0.0, 0.0, math.pi / 2),
    ("turn_right", 0.0, 0.0, -math.pi / 2),
    ("turn_around", 0.0, 0.0, math.pi),
    ("forward_turn_left", 0.3, 0.0, math.pi / 2),
    ("forward_turn_right", 0.3, 0.0, -math.pi / 2),
)


@dataclass(frozen=True)
class Sim2simConfig:
    mujoco_model_path: str
    duration: float = 12.0
    dt: float = 0.001
    decimation: int = 10
    cycle_time: float = 0.4
    frame_stack: int = 15
    num_single_obs: int = 47
    num_actions: int = 12
    action_scale: float = 0.25
    clip_observations: float = 18.0
    clip_actions: float = 18.0
    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 1.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    quat_scale: float = 1.0
    fall_height: float = 0.3
    fall_projected_gravity: float = 0.8
    base_contact_force: float = 1.0

    @property
    def num_observations(self):
        return self.frame_stack * self.num_single_obs

    @property
    def policy_dt(self):
        return self.dt * self.decimation


KP = np.array([40, 20, 20, 40, 40, 20] * 2, dtype=np.float64)
KD = np.array([0.6, 0.4, 0.4, 0.6, 0.6, 0.4] * 2, dtype=np.float64)
TAU_LIMIT = np.full(12, 17.85, dtype=np.float64)


def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quaternion_to_rotation_matrix(quat_xyzw):
    x, y, z, w = quat_xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_to_euler(quat_xyzw):
    x, y, z, w = quat_xyzw
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def swap_leg_halves(values):
    values = np.asarray(values, dtype=np.float64)
    return np.concatenate((values[6:12], values[:6]))


def get_state(data, num_actions):
    # MuJoCo framequat is [w, x, y, z]; policy helpers use [x, y, z, w].
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float64)
    rotation = quaternion_to_rotation_matrix(quat)
    body_velocity = rotation.T @ data.qvel[:3].astype(np.float64)
    body_angular_velocity = data.sensor("angular-velocity").data.astype(np.float64).copy()
    projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    euler = quaternion_to_euler(quat)

    # Preserve the deployment mapping used by the original sim2sim script.
    q = swap_leg_halves(data.qpos[-num_actions:])
    dq = swap_leg_halves(data.qvel[-num_actions:])
    # The MJCF uses three slide joints under a body with a non-zero initial pos,
    # so qpos[:3] is displacement rather than the absolute base position.
    position = data.sensor("position").data.astype(np.float64).copy()
    return q, dq, position, body_velocity, body_angular_velocity, projected_gravity, euler


def pd_control(target_q, q, target_dq, dq):
    return np.clip((target_q - q) * KP + (target_dq - dq) * KD, -TAU_LIMIT, TAU_LIMIT)


def base_has_terminating_contact(model, data, force_threshold):
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        if body1 != base_body_id and body2 != base_body_id:
            continue
        mujoco.mj_contactForce(model, data, contact_id, force)
        if np.linalg.norm(force[:3]) > force_threshold:
            return True
    return False


def termination_reason(model, data, state, cfg):
    _, _, position, _, _, projected_gravity, _ = state
    if base_has_terminating_contact(model, data, cfg.base_contact_force):
        return "base_contact"
    if position[2] < cfg.fall_height:
        return "base_height"
    if abs(projected_gravity[0]) > cfg.fall_projected_gravity:
        return "orientation_x"
    if abs(projected_gravity[1]) > cfg.fall_projected_gravity:
        return "orientation_y"
    return ""


def build_observation(state, previous_action, command, phase, cfg):
    q, dq, _, _, omega, _, euler = state
    obs = np.zeros(cfg.num_single_obs, dtype=np.float32)
    obs[0] = math.sin(2 * math.pi * phase)
    obs[1] = math.cos(2 * math.pi * phase)
    obs[2] = command[0] * cfg.lin_vel_scale
    obs[3] = command[1] * cfg.lin_vel_scale
    obs[4] = command[2] * cfg.ang_vel_scale
    obs[5:17] = q * cfg.dof_pos_scale
    obs[17:29] = dq * cfg.dof_vel_scale
    obs[29:41] = previous_action
    obs[41:44] = omega * cfg.ang_vel_scale
    obs[44:47] = euler * cfg.quat_scale
    return np.clip(obs, -cfg.clip_observations, cfg.clip_observations)


def make_trajectory_fields(num_actions):
    fields = [
        "case", "step", "time", "command_vx", "command_vy", "command_yaw_rate",
        "target_heading", "position_x", "position_y", "position_z", "velocity_x",
        "velocity_y", "velocity_z", "angular_velocity_x", "angular_velocity_y",
        "angular_velocity_z", "roll", "pitch", "yaw", "heading_error", "terminated",
        "termination_reason",
    ]
    for prefix in ("q", "dq", "action", "torque"):
        fields.extend(f"{prefix}_{index}" for index in range(num_actions))
    return fields


def make_trajectory_row(case_name, step, command, target_heading, state, action, torque,
                        terminated, reason, cfg):
    q, dq, position, velocity, omega, _, euler = state
    row = {
        "case": case_name,
        "step": step,
        "time": step * cfg.policy_dt,
        "command_vx": command[0],
        "command_vy": command[1],
        "command_yaw_rate": command[2],
        "target_heading": target_heading,
        "position_x": position[0],
        "position_y": position[1],
        "position_z": position[2],
        "velocity_x": velocity[0],
        "velocity_y": velocity[1],
        "velocity_z": velocity[2],
        "angular_velocity_x": omega[0],
        "angular_velocity_y": omega[1],
        "angular_velocity_z": omega[2],
        "roll": euler[0],
        "pitch": euler[1],
        "yaw": euler[2],
        "heading_error": abs(wrap_to_pi(target_heading - euler[2])),
        "terminated": int(terminated),
        "termination_reason": reason,
    }
    for prefix, values in (("q", q), ("dq", dq), ("action", action), ("torque", torque)):
        row.update({f"{prefix}_{index}": value for index, value in enumerate(values)})
    return row


def create_viewer(model, data, headless):
    if headless:
        return None
    try:
        import glfw
        import mujoco_viewer
    except ImportError as exc:
        raise RuntimeError(
            "GUI mode requires glfw and mujoco-python-viewer; use --headless for data-only evaluation."
        ) from exc
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")
    return mujoco_viewer.MujocoViewer(model, data)


def viewer_is_open(viewer):
    import glfw

    return not glfw.window_should_close(viewer.window)


def create_video_writer(model, output_path, cfg):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("--record-video requires opencv-python.") from exc
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / cfg.policy_dt, (width, height)
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"Cannot open video output: {output_path}")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    camera.distance = 1.2
    camera.azimuth = 135
    camera.elevation = -20
    return cv2, renderer, writer, camera


def evaluate_case(policy, cfg, case, trajectory_writer, headless, record_video, video_dir):
    case_name, command_vx, command_vy, target_heading = case
    model = mujoco.MjModel.from_xml_path(cfg.mujoco_model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    viewer = create_viewer(model, data, headless)
    video = create_video_writer(model, video_dir / f"{case_name}.mp4", cfg) if record_video else None

    history = deque(
        (np.zeros(cfg.num_single_obs, dtype=np.float32) for _ in range(cfg.frame_stack)),
        maxlen=cfg.frame_stack,
    )
    previous_action = np.zeros(cfg.num_actions, dtype=np.float64)
    target_q = np.zeros(cfg.num_actions, dtype=np.float64)
    target_dq = np.zeros(cfg.num_actions, dtype=np.float64)
    last_torque = np.zeros(cfg.num_actions, dtype=np.float64)
    errors = np.zeros(4, dtype=np.float64)
    steps = 0
    reason = ""
    max_steps = int(math.ceil(cfg.duration / cfg.policy_dt))

    try:
        for policy_step in tqdm(range(max_steps), desc=case_name):
            if viewer is not None and not viewer_is_open(viewer):
                reason = "viewer_closed"
                break

            state = get_state(data, cfg.num_actions)
            current_yaw = state[-1][2]
            command_yaw = np.clip(0.5 * wrap_to_pi(target_heading - current_yaw), -1.0, 1.0)
            command = np.array([command_vx, command_vy, command_yaw], dtype=np.float64)
            phase = policy_step * cfg.policy_dt / cfg.cycle_time
            history.append(build_observation(state, previous_action, command, phase, cfg))
            policy_input = np.concatenate(history).reshape(1, cfg.num_observations)
            with torch.no_grad():
                action = policy(torch.from_numpy(policy_input))[0].detach().cpu().numpy()
            action = np.clip(action, -cfg.clip_actions, cfg.clip_actions).astype(np.float64)
            target_q[:] = action * cfg.action_scale

            for _ in range(cfg.decimation):
                q, dq, *_ = get_state(data, cfg.num_actions)
                last_torque = pd_control(target_q, q, target_dq, dq)
                data.ctrl[:] = swap_leg_halves(last_torque)
                mujoco.mj_step(model, data)

            state = get_state(data, cfg.num_actions)
            _, _, _, velocity, omega, _, euler = state
            reason = termination_reason(model, data, state, cfg)
            terminated = bool(reason)
            errors += (
                abs(velocity[0] - command[0]),
                abs(velocity[1] - command[1]),
                abs(omega[2] - command[2]),
                abs(wrap_to_pi(target_heading - euler[2])),
            )
            steps += 1
            trajectory_writer.writerow(
                make_trajectory_row(
                    case_name, steps, command, target_heading, state, action, last_torque,
                    terminated, reason, cfg,
                )
            )
            previous_action[:] = action

            if viewer is not None:
                viewer.render()
                time.sleep(cfg.policy_dt)
            if video is not None:
                cv2, renderer, writer, camera = video
                renderer.update_scene(data, camera=camera)
                writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            if terminated:
                break
    finally:
        if viewer is not None:
            viewer.close()
            import glfw

            glfw.terminate()
        if video is not None:
            _, renderer, writer, _ = video
            writer.release()
            renderer.close()

    timeout = int(steps == max_steps and not reason)
    divisor = max(steps, 1)
    return {
        "case": case_name,
        "env_id": 0,
        "command_vx": command_vx,
        "command_vy": command_vy,
        "target_heading": target_heading,
        "steps": steps,
        "seconds": steps * cfg.policy_dt,
        "timeout": timeout,
        "termination_reason": reason or "timeout",
        "vx_mae": errors[0] / divisor,
        "vy_mae": errors[1] / divisor,
        "yaw_rate_mae": errors[2] / divisor,
        "heading_mae": errors[3] / divisor,
    }


def resolve_run_dir(load_model, explicit_run_dir):
    if explicit_run_dir:
        run_dir = Path(explicit_run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return run_dir

    model_path = Path(load_model).expanduser().resolve()
    for parent in model_path.parents:
        if any(parent.glob("model_*.pt")):
            return parent

    # Exported policies do not encode their source run, so use the newest run by default.
    if model_path.parent.name == "policies" and model_path.parent.parent.name == "exported":
        experiment_dir = model_path.parent.parent.parent
        candidates = [
            path for path in experiment_dir.iterdir()
            if path.is_dir() and path.name != "exported" and any(path.glob("model_*.pt"))
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)

    raise ValueError(
        "Cannot infer the training run from --load_model. Pass --run_dir, for example "
        "--run_dir logs/gru/Nov08_22-06-57_v2."
    )


def create_output_dir(args):
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        run_dir = resolve_run_dir(args.load_model, args.run_dir)
        output_dir = run_dir / "sim2sim" / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{args.seed}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_report(output_dir, args, cfg, rows):
    with (output_dir / "report.txt").open("w", encoding="utf-8") as report:
        report.write(f"Policy: {Path(args.load_model).resolve()}\n")
        report.write(f"MuJoCo model: {Path(cfg.mujoco_model_path).resolve()}\n")
        report.write(f"Seed: {args.seed}\nEngine: MuJoCo {mujoco.__version__}\nEnvironments: 1\n")
        report.write(f"Episode limit: {cfg.duration}s; policy dt: {cfg.policy_dt}s\n")
        report.write("Each command starts from a fresh MuJoCo model and data state.\n")
        report.write("No domain randomization, additional observation noise, action delay, or scheduled pushes are applied.\n")
        report.write("Sensor noise declared in the MuJoCo XML remains active when supported by the installed version.\n")
        report.write("Timeout=1 means the episode reached its time limit; timeout=0 means early termination.\n")
        report.write("Per-policy-step state, action, and torque data: trajectory.csv\n")
        report.write(f"Videos: {output_dir / 'videos' if args.record_video else 'disabled'}\n\n")
        for row in rows:
            report.write(
                f"{row['case']}: vx={row['command_vx']}, vy={row['command_vy']}, "
                f"target_heading={row['target_heading']:.3f}; timeout={row['timeout']}, "
                f"seconds={row['seconds']:.3f}, termination={row['termination_reason']}, "
                f"vx_mae={row['vx_mae']:.3f}, vy_mae={row['vy_mae']:.3f}, "
                f"yaw_rate_mae={row['yaw_rate_mae']:.3f}, heading_mae={row['heading_mae']:.3f}\n"
            )


def evaluate(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_path = Path(args.load_model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Policy does not exist: {model_path}")

    cfg = Sim2simConfig(
        mujoco_model_path=str(Path(args.mujoco_model).expanduser().resolve()),
        duration=args.duration,
    )
    output_dir = create_output_dir(args)
    video_dir = output_dir / "videos"
    if args.record_video:
        video_dir.mkdir()

    policy = torch.jit.load(str(model_path), map_location="cpu")
    policy.eval()
    selected_names = set(args.cases) if args.cases else None
    cases = [case for case in COMMANDS if selected_names is None or case[0] in selected_names]
    episode_fields = [
        "case", "env_id", "command_vx", "command_vy", "target_heading", "steps",
        "seconds", "timeout", "termination_reason", "vx_mae", "vy_mae",
        "yaw_rate_mae", "heading_mae",
    ]
    rows = []
    try:
        with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as trajectory_file:
            trajectory_writer = csv.DictWriter(
                trajectory_file, fieldnames=make_trajectory_fields(cfg.num_actions)
            )
            trajectory_writer.writeheader()
            for case in cases:
                row = evaluate_case(
                    policy, cfg, case, trajectory_writer, args.headless, args.record_video, video_dir
                )
                rows.append(row)
                trajectory_file.flush()

        with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as episodes_file:
            writer = csv.DictWriter(episodes_file, fieldnames=episode_fields)
            writer.writeheader()
            writer.writerows(rows)
        write_report(output_dir, args, cfg, rows)
        metadata = {
            "policy": str(model_path),
            "run_dir": str(resolve_run_dir(args.load_model, args.run_dir)) if not args.output_dir else None,
            "output_dir": str(output_dir),
            "seed": args.seed,
            "cases": [case[0] for case in cases],
            "config": asdict(cfg),
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    except Exception:
        print(f"Evaluation failed; partial results remain in: {output_dir}")
        raise

    print(f"Sim2sim evaluation saved to: {output_dir}")
    return output_dir


def parse_args():
    default_policy = Path(LEGGED_GYM_ROOT_DIR) / "logs/gru/exported/policies/policy_torch.pt"
    default_mujoco_model = (
        Path(LEGGED_GYM_ROOT_DIR)
        / "resources/robots/pi_12dof_release_v1/mjcf/pi_12dof_release_v1.xml"
    )
    parser = argparse.ArgumentParser(description="Evaluate an exported policy in MuJoCo.")
    parser.add_argument("--load_model", default=str(default_policy), help="TorchScript policy path.")
    parser.add_argument(
        "--run_dir",
        help="Training run directory. Results are saved under <run_dir>/sim2sim/<timestamp>.",
    )
    parser.add_argument("--output_dir", help="Exact output directory override, mainly for smoke tests.")
    parser.add_argument("--mujoco_model", default=str(default_mujoco_model), help="MuJoCo XML path.")
    parser.add_argument("--duration", type=float, default=12.0, help="Maximum seconds per case.")
    parser.add_argument("--seed", type=int, default=123145)
    parser.add_argument(
        "--cases", nargs="+", choices=[case[0] for case in COMMANDS],
        help="Cases to run; omit to run all play.py command cases.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without a GLFW viewer.")
    parser.add_argument("--record-video", action="store_true", help="Save one MP4 per command case.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
