import os
import argparse
import ale_py
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
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
    # opt: mejor prueba de Optuna (#4, score -11.99 a 400k). La #14, casi
    # idéntica (lr 2.7e-4, ent 3.8e-4), quedó segunda con -12.52.
    "opt": dict(
        n_envs=8,
        ppo=dict(
            n_steps=128,
            batch_size=256,
            n_epochs=10,
            learning_rate=2.444e-4,
            clip_range=0.2,
            target_kl=0.02,
            ent_coef=6.956e-4,
        ),
    ),
    # opt_shape (Exp. 4): opt + reward shaping basado en potenciales con la RAM
    "opt_shape": dict(
        n_envs=8,
        shaping=True,
        ppo=dict(
            n_steps=128,
            batch_size=256,
            n_epochs=10,
            learning_rate=2.444e-4,
            clip_range=0.2,
            target_kl=0.02,
            ent_coef=6.956e-4,
        ),
    ),
    # opt_shape2 (Exp. 4b): opt + shaping que anticipa la trayectoria
    "opt_shape2": dict(
        n_envs=8,
        shaping="v2",
        ppo=dict(
            n_steps=128,
            batch_size=256,
            n_epochs=10,
            learning_rate=2.444e-4,
            clip_range=0.2,
            target_kl=0.02,
            ent_coef=6.956e-4,
        ),
    ),
}


# ============================================================
# Reward shaping basado en potenciales (Ng, Harada y Russell, 1999)
# ============================================================
# RAM de Pong verificada con check_ram.py:
#   ram[49] = pelota x (crece cuando viene hacia nuestra pala, a la derecha)
#   ram[54] = pelota y (0 cuando no hay pelota en juego)
#   ram[51] = nuestra pala y (mismo sistema de coordenadas que la pelota)
#
# Potencial: cuando la pelota viene hacia nosotros,
#   phi(s) = -LAMBDA * |pelota_y - pala_y| / 160      (0 en otro caso)
# Recompensa extra: F = GAMMA * phi(s') - phi(s). Al ser basada en potencial
# no cambia la política óptima (se cancela a lo largo del episodio). Solo se
# usa en ENTRENAMIENTO: la evaluación y el Monitor ven la recompensa real.

SHAPING_LAMBDA = 0.5
SHAPING_GAMMA = 0.99   # debe coincidir con el gamma de PPO (0.99 por defecto)


class PongShaping(gym.Wrapper):
    def __init__(self, env, lam=SHAPING_LAMBDA, gamma=SHAPING_GAMMA):
        super().__init__(env)
        self.lam, self.gamma = lam, gamma
        self.prev_phi = 0.0
        self.last_x = None

    def _phi(self):
        ram = self.env.unwrapped.ale.getRAM()
        bx, by, py = int(ram[49]), int(ram[54]), int(ram[51])
        if bx == 0 or by == 0:          # no hay pelota en juego
            self.last_x = None
            return 0.0
        coming = self.last_x is not None and bx > self.last_x
        self.last_x = bx
        if not coming:
            return 0.0
        return -self.lam * abs(by - py) / 160.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_x = None
        self.prev_phi = self._phi()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi = self._phi()
        if reward != 0 or terminated or truncated:
            phi = 0.0                    # potencial 0 al acabar el punto
        shaped = reward + self.gamma * phi - self.prev_phi
        self.prev_phi = phi
        return obs, shaped, terminated, truncated, info


def shaped_atari_wrapper(env, **kwargs):
    """AtariWrapper normal + shaping encima (el Monitor queda debajo y
    registra la recompensa real)."""
    return PongShaping(AtariWrapper(env, **kwargs))


# ------------------------------------------------------------
# Shaping v2: anticipar la trayectoria (Exp. 4b)
# ------------------------------------------------------------
# Observación al ver jugar al agente: pierde casi todos los puntos con bolas
# de ángulo muy inclinado (|dy| alto). La velocidad horizontal es casi
# constante (4 px/paso), pero con mucho |dy| la pelota rebota en las paredes
# y no da tiempo a seguirla: hay que moverse hacia donde VA a llegar.
# El shaping v1 premiaba alinearse con la posición ACTUAL, lo contrario.
#
# Geometría medida en la RAM: nuestra pala en x ~ 188, paredes en y ~ 44 y
# ~ 207; la pala devuelve mejor con pala_y ~ pelota_y - 3. El predictor
# acierta el punto de llegada con un error mediano de 2-5 px.
#
# Potencial, cuando la pelota viene hacia nosotros:
#   phi = -LAMBDA2 * w * |y_llegada - 3 - pala_y| / 163,  w = min(1 + |dy|/8, 2)
# (más peso a las bolas inclinadas, que son las que se pierden). Sigue siendo
# basado en potenciales: F = GAMMA * phi(s') - phi(s).

SHAPING2_LAMBDA = 0.3
PADDLE_X, Y_MIN, Y_MAX = 188, 44, 207


def predict_arrival_y(bx, by, dx, dy):
    """Altura a la que llegará la pelota a nuestra pala, con rebotes."""
    t = (PADDLE_X - bx) / dx
    span = Y_MAX - Y_MIN
    y = (by + dy * t - Y_MIN) % (2 * span)
    if y > span:
        y = 2 * span - y
    return Y_MIN + y


class PongShapingPredictive(gym.Wrapper):
    def __init__(self, env, lam=SHAPING2_LAMBDA, gamma=SHAPING_GAMMA):
        super().__init__(env)
        self.lam, self.gamma = lam, gamma
        self.prev_phi = 0.0
        self.last = None

    def _phi(self):
        ram = self.env.unwrapped.ale.getRAM()
        bx, by, py = int(ram[49]), int(ram[54]), int(ram[51])
        if bx == 0 or by == 0:              # no hay pelota en juego
            self.last = None
            return 0.0
        last, self.last = self.last, (bx, by)
        if last is None:
            return 0.0
        dx, dy = bx - last[0], by - last[1]
        if dx <= 0 or bx >= PADDLE_X:       # se aleja o ya ha llegado
            return 0.0
        y_arr = predict_arrival_y(bx, by, dx, dy)
        w = min(1.0 + abs(dy) / 8.0, 2.0)
        return -self.lam * w * abs(y_arr - 3 - py) / (Y_MAX - Y_MIN)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last = None
        self.prev_phi = self._phi()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi = self._phi()
        if reward != 0 or terminated or truncated:
            phi = 0.0
        shaped = reward + self.gamma * phi - self.prev_phi
        self.prev_phi = phi
        return obs, shaped, terminated, truncated, info


def predictive_atari_wrapper(env, **kwargs):
    return PongShapingPredictive(AtariWrapper(env, **kwargs))


SHAPING_WRAPPERS = {
    True: shaped_atari_wrapper,       # v1 (opt_shape)
    "v1": shaped_atari_wrapper,
    "v2": predictive_atari_wrapper,   # v2 (opt_shape2)
}

parser = argparse.ArgumentParser()
parser.add_argument("--exp", choices=EXPERIMENTS.keys(), default="exp1")
parser.add_argument("--steps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default=None,
                    help="Continuar desde un modelo .zip (p. ej. models/opt/best_model/best_model)")
parser.add_argument("--name", type=str, default=None,
                    help="Nombre del run (por defecto, el del experimento)")
parser.add_argument("--lr_decay", type=float, default=None,
                    help="Solo con --resume: learning_rate que baja linealmente "
                         "desde este valor hasta 0 durante --steps (afinado)")
args = parser.parse_args()

# Nombre del run: con la semilla por defecto (42) se mantiene el de siempre;
# con otra semilla se añade el sufijo, para no sobrescribir modelos ni logs.
RUN = args.exp if args.seed == 42 else f"{args.exp}_s{args.seed}"
if args.name:
    RUN = args.name

EXP = EXPERIMENTS[args.exp]


# ============================================================
# Configuration
# ============================================================

ENV_ID = "ALE/Pong-v5"
MAX_TIMESTEPS = args.steps
MODELS_DIR = f"models/{RUN}"
LOG_DIR = "logs"
SEED = args.seed
EVAL_EVERY = 25_000          # pasos totales entre evaluaciones
N_EVAL_EPISODES = 10         # 5 era demasiado ruidoso

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

gym.register_envs(ale_py)


# ============================================================
# Environment (mismo preprocesado que utils.play_agent)
# ============================================================

WRAPPER_KWARGS = {
    "frame_skip": 4,
    "screen_size": 84,
    "terminal_on_life_loss": False,
    "clip_reward": False,
}


def create_env(seed, n_envs=1, shaping=False):
    if shaping:
        # Igual que make_atari_env, pero con el shaping encima del AtariWrapper
        env = make_vec_env(
            ENV_ID,
            n_envs=n_envs,
            seed=seed,
            wrapper_class=SHAPING_WRAPPERS[shaping],
            wrapper_kwargs=WRAPPER_KWARGS,
            env_kwargs={"frameskip": 1},
            vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
        )
    else:
        env = make_atari_env(
            ENV_ID,
            n_envs=n_envs,
            seed=seed,
            env_kwargs={"frameskip": 1},
            wrapper_kwargs=WRAPPER_KWARGS,
            vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
        )
    return VecFrameStack(env, n_stack=4)


if __name__ == "__main__":
    env = create_env(SEED, EXP["n_envs"], shaping=EXP.get("shaping", False))
    eval_env = create_env(SEED + 1000)   # la evaluación nunca lleva shaping

    # EvalCallback cuenta llamadas a step() del VecEnv, no pasos totales
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/best_model",
        log_path=f"{LOG_DIR}/eval_{RUN}",
        eval_freq=max(EVAL_EVERY // EXP["n_envs"], 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    if args.resume:
        # Continuar un modelo ya entrenado con sus mismos hiperparámetros
        model = PPO.load(args.resume, env=env, tensorboard_log=LOG_DIR,
                         seed=SEED, verbose=1)
        print(f"Continuando desde {args.resume} ({model.num_timesteps} pasos previos)")
        if args.lr_decay:
            # Afinado: lr de lr_decay -> 0 a lo largo de los --steps nuevos.
            # Al continuar, SB3 calcula el progreso sobre (pasos previos +
            # nuevos), así que se reescala para que empiece en lr_decay.
            lr0, n0 = args.lr_decay, model.num_timesteps
            p0 = MAX_TIMESTEPS / (n0 + MAX_TIMESTEPS)
            model.lr_schedule = lambda p: lr0 * max(0.0, min(1.0, p / p0))
            print(f"Afinado: learning_rate {lr0} -> 0 en {MAX_TIMESTEPS} pasos")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            seed=SEED,
            **EXP["ppo"],
        )

    print(f"Experimento: {RUN} | seed={SEED} | n_envs={EXP['n_envs']} | "
          f"shaping={EXP.get('shaping', False)} | {EXP['ppo']}")

    model.learn(
        total_timesteps=MAX_TIMESTEPS,
        tb_log_name=RUN,
        callback=eval_callback,
        reset_num_timesteps=not args.resume,
    )
    model.save(f"{MODELS_DIR}/final_model")

    # Evaluación final del mejor modelo con 20 partidas
    # El profesor carga con PPO.load y en sus scripts evalúa a veces con
    # deterministic=False, así que medimos los dos modos.
    best = PPO.load(f"{MODELS_DIR}/best_model/best_model", env=eval_env)
    print()
    print("=" * 50)
    print(f"FINAL EVALUATION ({RUN}, best_model, 20 episodios)")
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