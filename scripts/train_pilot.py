"""Train a drone pilot policy with PPO in the MuJoCo simulator.

Tasks:
    avoidance  fly through the obstacle course to the finish line (src/sim_env.py)
    mission    find a person / vehicle / drone and fly up to it (src/mission_env.py)

Examples:
    python scripts/train_pilot.py --task avoidance                  # 3M steps
    python scripts/train_pilot.py --task mission                    # 3M steps
    python scripts/train_pilot.py --task mission --timesteps 200000 # quick trial
    python scripts/train_pilot.py --task mission --resume runs/mission/mission/best_model.zip

Progress: tensorboard --logdir runs/<task>/tensorboard
Ctrl+C stops training and still saves the model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.mission_env import DroneMissionEnv  # noqa: E402
from src.sim_env import DroneAvoidanceEnv  # noqa: E402

TASKS = {
    # task: (environment, evaluation settings: fixed, fairly hard, comparable over time)
    "avoidance": (DroneAvoidanceEnv, {"min_obstacles": 30, "max_obstacles": 30}),
    "mission": (DroneMissionEnv, {"min_obstacles": 20, "max_obstacles": 20}),
}


def new_run_dir(runs_dir: Path, name: str) -> Path:
    run_dir = runs_dir / name
    suffix = 2
    while run_dir.exists():
        run_dir = runs_dir / f"{name}-{suffix}"
        suffix += 1
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, default="avoidance")
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--envs", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel simulator processes (default: up to 8).")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'.")
    parser.add_argument("--name", help="Run folder name under runs/<task> (default: the task name).")
    parser.add_argument("--resume", type=Path, help="Continue training from a saved .zip model.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env_class, eval_kwargs = TASKS[args.task]
    runs_dir = ROOT / "runs" / args.task
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = new_run_dir(runs_dir, args.name or args.task)
    run_dir.mkdir(parents=True)
    print(f"Task: {args.task}\nRun folder: {run_dir}\nDevice: {device}, simulators: {args.envs}", flush=True)

    env = make_vec_env(env_class, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(env_class, n_envs=1, seed=args.seed + 10_000, env_kwargs=eval_kwargs)

    if args.resume:
        model = PPO.load(args.resume, env=env, device=device)
        print(f"Resumed from {args.resume} at {model.num_timesteps} steps")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=512,
            batch_size=512,
            n_epochs=5,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            tensorboard_log=str(runs_dir / "tensorboard"),
            device=device,
            seed=args.seed,
            verbose=1,
        )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),
            eval_freq=max(50_000 // args.envs, 1),
            n_eval_episodes=20,
            deterministic=True,
        ),
        CheckpointCallback(save_freq=max(250_000 // args.envs, 1), save_path=str(run_dir / "checkpoints")),
    ]

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=run_dir.name,
            reset_num_timesteps=args.resume is None,
        )
    except KeyboardInterrupt:
        print("\nStopped by user, saving model...")
    finally:
        model.save(run_dir / "final_model")
        env.close()
        eval_env.close()
    print(f"Saved: {run_dir / 'final_model.zip'} (best: {run_dir / 'best_model.zip'})")


if __name__ == "__main__":
    main()
