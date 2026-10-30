import os
import ale_py
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


# ============================================================
# Configuration
# ============================================================

ENV_ID = "ALE/Pong-v5"
ALGORITHM = "PPO"
MAX_TIMESTEPS = 1_000_000
MODELS_DIR = f"models/{ALGORITHM}"
LOG_DIR = "logs"
SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# Register ALE environments in Gymnasium
gym.register_envs(ale_py)


# ============================================================
# Environment
# ============================================================

def create_env(seed):
    """
    Creates an Atari Pong environment.

    ALE/Pong-v5 normally performs 4 frames per action.
    Stable-Baselines3's AtariWrapper also performs frame skipping,
    so we create the underlying ALE environment with frameskip=1
    and let the SB3 wrapper handle it.
    """

    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=seed,
        env_kwargs={
            "frameskip": 1,
        },
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "terminal_on_life_loss": False,
            "clip_reward": False,
        },
    )

    # Stack 4 frames so the agent can infer movement/velocity
    env = VecFrameStack(env,n_stack=4)

    return env


env = create_env(SEED)
eval_env = create_env(SEED + 1000)


# ============================================================
# Evaluation callback
# ============================================================

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{MODELS_DIR}/best_model",
    log_path=f"{LOG_DIR}/eval",
    eval_freq=25_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)


# ============================================================
# Agent
# ============================================================

model = PPO(
    "CnnPolicy",
    env,
    # Using 1000 makes MAX_TIMESTEPS=1,000,000 an exact
    # multiple of the rollout size.
    n_steps=1000,
    batch_size=100,
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=SEED,
)


# ============================================================
# Training
# ============================================================

model.learn(
    total_timesteps=MAX_TIMESTEPS,
    reset_num_timesteps=True,
    tb_log_name=ALGORITHM,
    callback=eval_callback,
)

model.save(
    f"{MODELS_DIR}/final_model"
)


# ============================================================
# Evaluation
# ============================================================

mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True,
)

print()
print("=" * 50)
print("FINAL EVALUATION")
print("=" * 50)
print(f"Steps:       {MAX_TIMESTEPS}")
print(f"Mean reward: {mean_reward:.2f}")
print(f"Std reward:  {std_reward:.2f}")
print("=" * 50)


env.close()
eval_env.close()