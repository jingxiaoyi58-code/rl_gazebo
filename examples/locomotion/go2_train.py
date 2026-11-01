import argparse
import math
import os
import pickle
from importlib import metadata

# Select the validated solver before Genesis/Quadrants reads its environment settings.
os.environ["QD_PERFDISPATCH_FORCE"] = "func_solve_body:func_solve_body_monolith"

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

import genesis as gs
import torch

from go2_env import Go2Env
from go2_runner import Go2Runner, entropy_coef_at_iteration


def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    # reward_cfg = {
    # "tracking_sigma": 0.25,
    # "base_height_target": 0.3,
    # "feet_height_target": 0.075,
    # "reward_scales": {
    # "tracking_lin_vel": 1.0,
    # "tracking_ang_vel": 0.2,
    # "lin_vel_z": -1.0,
    # "base_height": -50.0,
    # "action_rate": -0.005,
    # "similar_to_default": -0.1,
    # },
    # }
    reward_cfg = {
        # 如果环境使用常见的指数速度跟踪奖励，
        # 适当增大此值，可以让速度误差较大时仍获得一定奖励。
        "tracking_sigma": 0.5,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "joint_target_limit_margin": 0.05,  # 关节目标角度在物理限位两端各留 5% 范围余量
        "feet_contact_force_threshold": 5.0,  # 足端竖直接触力超过 5 N 才计为支撑
        "feet_usage_target": 0.3,  # 每只脚近期的平滑接触占比低于 30% 时产生惩罚
        "feet_usage_time_constant": 0.4,  # 接触占比的平滑时间常数，单位秒
        "feet_usage_min_command": 0.2,  # 目标水平速度超过 0.2 m/s 时启用支撑不足惩罚
        "reward_scales": {
            "tracking_lin_vel": 3.0,  # 原来 1.0：提高跟随目标线速度的重要性
            "tracking_ang_vel": 0.2,  # 跟踪转向角速度指令，支持左右转弯
            "lin_vel_z": -1.0,  # 加强上下速度惩罚，减少蹦跳
            "base_height": -30.0,  # 约束机身偏离 0.3 m 目标高度
            "action_rate": -0.005,  # 保留动作平滑惩罚，避免关节剧烈抖动
            "similar_to_default": -0.1,  # 惩罚关节偏离默认姿态，保留当前设置
            "joint_target_limits": -0.5,  # 惩罚关节目标角度超出带余量的限位范围
            "feet_usage": -0.5,  # 辅助防止长期吊腿；不能单靠此项规定步态
            "diagonal_support": 1.5,  # 奖励一组对角脚支撑、另一组离地，鼓励小跑步态
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.8, 0.8],  # 初始前后速度范围，m/s
        "lin_vel_y_range": [-0.3, 0.3],  # 初始左右侧移速度范围，m/s
        "ang_vel_range": [-0.2, 0.2],  # 初始左右转向角速度范围，rad/s
        "curriculum": {
            "enabled": True,
            "final_lin_vel_x_range": [-1.5, 1.5],  # 课程末期前后速度范围
            "final_lin_vel_y_range": [-0.5, 0.5],  # 课程末期侧移速度范围
            "final_ang_vel_range": [-0.5, 0.5],  # 课程末期转向角速度范围
            "num_levels": 10,  # 初始为第 0 级，成功升级 10 次后到达最终范围
            "update_interval_s": 20.0,  # 每 20 秒仿真时间检查一次，不是现实等待时间
            "linear_tracking_threshold": 0.8,  # 未加权的线速度跟踪奖励均值需达到 0.8
            "angular_tracking_threshold": 0.9,  # 未加权的转向跟踪奖励均值需达到 0.9
            "survival_threshold": 0.9,  # 本窗口内至少 90% 的环境没有摔倒或物理报错
        },
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--max-iterations", type=int, default=101, help="Number of new learning iterations to run")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--headless", action="store_true", help="Train without opening the visualization window")
    parser.add_argument("--resume", type=str, help="Checkpoint path to resume; loads cfgs.pkl from its directory")
    parser.add_argument("--entropy-coef", type=float, help="Override the final PPO entropy coefficient, including on resume")
    parser.add_argument("--entropy-decay-iterations", type=int, default=0,
                        help="Linearly decay from the current entropy coefficient over this many new iterations")
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_iterations < 1:
        parser.error("--num-envs and --max-iterations must be positive.")
    if args.entropy_coef is not None and (not math.isfinite(args.entropy_coef) or args.entropy_coef < 0):
        parser.error("--entropy-coef must be finite and nonnegative.")
    if args.entropy_decay_iterations < 0 or (args.entropy_decay_iterations and args.entropy_coef is None):
        parser.error("--entropy-decay-iterations must be nonnegative and requires --entropy-coef.")

    log_dir = f"logs/{args.exp_name}"
    if args.resume:
        # 续训沿用该检查点所属实验的配置，避免误用后来修改过的默认参数。
        args.resume = os.path.abspath(args.resume)
        if not os.path.isfile(args.resume):
            parser.error(f"Checkpoint does not exist: {args.resume}")
        with open(os.path.join(os.path.dirname(args.resume), "cfgs.pkl"), "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
        train_cfg["run_name"] = args.exp_name
        if os.path.exists(log_dir):
            parser.error("Resuming requires a new experiment name to preserve the original logs and checkpoints.")
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name)

    if args.entropy_coef is not None:
        # 先载入原实验配置，再显式覆盖；仅修改默认 get_train_cfg 不会影响续训。
        start = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            start = int(checkpoint["iter"]) + 1
            del checkpoint
        previous_schedule = train_cfg.get("entropy_schedule")
        initial = (entropy_coef_at_iteration(previous_schedule, start) if previous_schedule
                   else train_cfg["algorithm"]["entropy_coef"])
        train_cfg["algorithm"]["entropy_coef"] = args.entropy_coef
        if args.entropy_decay_iterations:
            train_cfg["entropy_schedule"] = {
                "initial_coef": initial, "final_coef": args.entropy_coef,
                "start_iteration": start, "end_iteration": start + args.entropy_decay_iterations - 1,
            }
        else:
            train_cfg.pop("entropy_schedule", None)

    os.makedirs(log_dir, exist_ok=not args.resume)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # Training throughput comes from the 4096 parallel environments, which need a GPU.
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
    )

    try:
        runner = Go2Runner(env, train_cfg, log_dir, device=gs.device)
        if args.resume:
            runner.load(args.resume, map_location=str(gs.device))
            # 检查点 iter 是已经完成的轮号；从下一轮开始，避免重复轮号。
            runner.current_learning_iteration += 1
        start_iteration = runner.current_learning_iteration
        final_iteration = start_iteration + args.max_iterations - 1
        if "entropy_schedule" in train_cfg:
            print(f"ENTROPY_SCHEDULE {train_cfg['entropy_schedule']}", flush=True)
        print(
            f"TRAIN_START num_envs={args.num_envs} headless={args.headless} "
            f"iterations={start_iteration}..{final_iteration} resume={args.resume}",
            flush=True,
        )
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        print(f"TRAIN_COMPLETE checkpoint={log_dir}/model_{runner.current_learning_iteration}.pt", flush=True)
    finally:
        env.scene.destroy()


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
