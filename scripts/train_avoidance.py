"""Train the depth-camera obstacle-avoidance policy with PPO in the MuJoCo course.

Examples:
    python scripts/train_avoidance.py                      # 3M steps, new run folder
    python scripts/train_avoidance.py --timesteps 200000   # quick trial
    python scripts/train_avoidance.py --resume runs/avoidance/avoidance/final_model.zip

Progress: tensorboard --logdir runs/avoidance/tensorboard
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
from src.sim_env import DroneAvoidanceEnv  # noqa: E402

RUNS_DIR = ROOT / "runs" / "avoidance"


def new_run_dir(name: str) -> Path:
    run_dir = RUNS_DIR / name
    suffix = 2
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{name}-{suffix}"
        suffix += 1
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--envs", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel simulator processes (default: up to 8).")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'.")
    parser.add_argument("--name", default="avoidance", help="Run folder name under runs/avoidance.")
    parser.add_argument("--resume", type=Path, help="Continue training from a saved .zip model.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = new_run_dir(args.name)
    run_dir.mkdir(parents=True)
    print(f"Run folder: {run_dir}\nDevice: {device}, simulators: {args.envs}")

    env = make_vec_env(DroneAvoidanceEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    # Evaluate on a fixed, fairly dense course so scores are comparable over time
    eval_env = make_vec_env(
        DroneAvoidanceEnv,
        n_envs=1,
        seed=args.seed + 10_000,
        env_kwargs={"min_obstacles": 30, "max_obstacles": 30},
    )

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
            tensorboard_log=str(RUNS_DIR / "tensorboard"),
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
