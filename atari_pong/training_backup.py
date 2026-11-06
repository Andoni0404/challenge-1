import os
import argparse
import ale_py
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


# ============================================================
# Experiments
# ============================================================
# baseline: configuración original del challenge
# exp1:     estabilizar (approx_kl muy alto en el baseline)
# exp2:     exp1 + exploración (entropía colapsaba a los 200k)
# exp3:     calibrar velocidad (exp1 demasiado lento, exp2 peor)

def linear(start):
    """Schedule lineal: start -> 0 a lo largo del entrenamiento."""
    return lambda progress_remaining: progress_remaining * start


EXPERIMENTS = {
    "baseline": dict(
        n_envs=1,
        ppo=dict(n_steps=1000, batch_size=100),
    ),
    "exp1": dict(
        n_envs=8,
        ppo=dict(
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            learning_rate=linear(2.5e-4),
            clip_range=linear(0.1),
        ),
    ),
    "exp2": dict(
        n_envs=8,
        ppo=dict(
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            learning_rate=linear(2.5e-4),
            clip_range=linear(0.1),
            ent_coef=0.01,
        ),
    ),
    # exp3: calibrar la velocidad. Exp1 era estable pero demasiado tímido
    # (approx_kl ~0.002 -> 0). Objetivo: KL ~0.01, con target_kl como freno.
    "exp3": dict(
        n_envs=8,
        ppo=dict(
            n_steps=128,
            batch_size=256,
            n_epochs=8,             # 32 actualizaciones por rollout (exp1: 16)
            learning_rate=3e-4,     # fijo: exp1 se congeló al llegar a 0
            clip_range=0.2,         # fijo
            target_kl=0.02,         # corta las épocas si el KL se dispara
            ent_coef=0.0,           # exp2 demostró que no hace falta
        ),
    ),
    # exp3b: exp3 colapsó la entropía (-0.2 a los 145k) antes de aprender.
    # ent_coef intermedio: 0 colapsa (exp3), 0.01 queda casi aleatorio (exp2).
    "exp3b": dict(
        n_envs=8,
        ppo=dict(
            n_steps=128,
            batch_size=256,
            n_epochs=8,
            learning_rate=3e-4,
            clip_range=0.2,
            target_kl=0.02,
            ent_coef=0.005,
        ),
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--exp", choices=EXPERIMENTS.keys(), default="exp1")
parser.add_argument("--steps", type=int, default=1_000_000)
args = parser.parse_args()

EXP = EXPERIMENTS[args.exp]


# ============================================================
# Configuration
# ============================================================

ENV_ID = "ALE/Pong-v5"
MAX_TIMESTEPS = args.steps
MODELS_DIR = f"models/{args.exp}"
LOG_DIR = "logs"
SEED = 42
EVAL_EVERY = 25_000          # pasos totales entre evaluaciones
N_EVAL_EPISODES = 10         # 5 era demasiado ruidoso

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

gym.register_envs(ale_py)


# ============================================================
# Environment (mismo preprocesado que utils.play_agent)
# ============================================================

def create_env(seed, n_envs=1):
    env = make_atari_env(
        ENV_ID,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"frameskip": 1},
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "terminal_on_life_loss": False,
            "clip_reward": False,
        },
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
    )
    return VecFrameStack(env, n_stack=4)


if __name__ == "__main__":
    env = create_env(SEED, EXP["n_envs"])
    eval_env = create_env(SEED + 1000)

    # EvalCallback cuenta llamadas a step() del VecEnv, no pasos totales
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/best_model",
        log_path=f"{LOG_DIR}/eval_{args.exp}",
        eval_freq=max(EVAL_EVERY // EXP["n_envs"], 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=SEED,
        **EXP["ppo"],
    )

    print(f"Experimento: {args.exp} | n_envs={EXP['n_envs']} | {EXP['ppo']}")

    model.learn(
        total_timesteps=MAX_TIMESTEPS,
        tb_log_name=args.exp,
        callback=eval_callback,
    )
    model.save(f"{MODELS_DIR}/final_model")

    # Evaluación final del mejor modelo con 20 partidas
    # El profesor carga con PPO.load y en sus scripts evalúa a veces con
    # deterministic=False, así que medimos los dos modos.
    best = PPO.load(f"{MODELS_DIR}/best_model/best_model", env=eval_env)
    print()
    print("=" * 50)
    print(f"FINAL EVALUATION ({args.exp}, best_model, 20 episodios)")
    print("=" * 50)
    print(f"Steps:       {MAX_TIMESTEPS}")
    for det in (True, False):
        mean_reward, std_reward = evaluate_policy(
            best, eval_env, n_eval_episodes=20, deterministic=det
        )
        modo = "determinista" if det else "estocástico "
        print(f"{modo}: {mean_reward:.2f} ± {std_reward:.2f}")
    print("=" * 50)

    env.close()
    eval_env.close()