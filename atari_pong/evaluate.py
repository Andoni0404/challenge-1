"""Evalúa un modelo guardado en modo determinista y estocástico.

Uso:
    python evaluate.py models/exp1/best_model/best_model --episodes 20
"""
import argparse
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=123)
args = parser.parse_args()

gym.register_envs(ale_py)

# Mismo preprocesado que utils.play_agent (el código del profesor)
env = make_atari_env(
    "ALE/Pong-v5",
    n_envs=1,
    seed=args.seed,
    env_kwargs={"frameskip": 1},
    wrapper_kwargs={
        "frame_skip": 4,
        "screen_size": 84,
        "terminal_on_life_loss": False,
        "clip_reward": False,
    },
)
env = VecFrameStack(env, n_stack=4)

model = PPO.load(args.model)  # sin env ni custom_objects, como el profesor

print(f"Modelo: {args.model} | {args.episodes} episodios")
for det in (True, False):
    mean, std = evaluate_policy(model, env, n_eval_episodes=args.episodes,
                                deterministic=det)
    print(f"  {'determinista' if det else 'estocástico '}: {mean:.2f} ± {std:.2f}")
env.close()