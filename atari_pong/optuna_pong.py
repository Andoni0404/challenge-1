"""
Búsqueda de hiperparámetros de PPO para Pong con Optuna.

- Sampler: GPSampler (optimización bayesiana con procesos gaussianos; buena con
  pocas pruebas y métricas ruidosas). HEBO se descartó: exige numpy<1.25.
- Pruning: PatientPruner(MedianPruner) con warmup de 150k pasos, para no matar
  configuraciones que arrancan tarde (como el Exp. 1), más una regla propia de
  colapso de entropía (lección del Exp. 3).
- Warm start: 3 configuraciones conocidas se prueban primero (enqueue_trial):
  Exp. 3b, Exp. 1 adaptado y RL Zoo.

Uso (con el venv activado, sin otro entrenamiento en marcha):
    pip install optuna scipy plotly
    nohup python optuna_pong.py --trials 15 --steps 400000 --hours 12 > logs/stdout/optuna.log 2>&1 &

El estudio se guarda en optuna/optuna_pong.db: si se corta, al relanzar continúa.
"""
import os
import argparse
import numpy as np
import ale_py
import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner, PatientPruner

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=15)
parser.add_argument("--steps", type=int, default=400_000)   # pasos por prueba
parser.add_argument("--hours", type=float, default=12.0)    # tiempo máximo total
parser.add_argument("--n_envs", type=int, default=8)
args = parser.parse_args()

gym.register_envs(ale_py)
ENV_ID = "ALE/Pong-v5"
SEED = 42
REPORT_EVERY = 50_000          # cada cuántos pasos se informa a Optuna
WARMUP = 150_000               # antes de esto no se poda nada
LOG_DIR = "logs/tensorboard/optuna"
os.makedirs("models/optuna", exist_ok=True)
os.makedirs("optuna", exist_ok=True)


def create_env(seed, n_envs):
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


def score(model):
    """Métrica: reward medio de las últimas partidas de entrenamiento
    + un pequeño término por duración que desempata cuando todas están en -21
    (partidas más largas = devuelve la pelota = progreso). Máx. ~+1 punto."""
    buf = model.ep_info_buffer
    if len(buf) == 0:
        return -21.0
    rew = float(np.mean([e["r"] for e in buf]))
    length = float(np.mean([e["l"] for e in buf]))
    return rew + min(length / 10_000, 1.0)


class OptunaCallback(BaseCallback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial
        self.next_report = REPORT_EVERY
        self.pruned = False
        self.reason = ""

    def _on_step(self):
        if self.num_timesteps < self.next_report:
            return True
        # Se informa siempre en 50k, 100k, 150k... (no en el paso exacto),
        # para que el MedianPruner compare todas las pruebas en el mismo paso.
        step = self.next_report
        self.next_report += REPORT_EVERY
        value = score(self.model)
        self.trial.report(value, step)

        # Regla propia: colapso de entropía sin haber aprendido (Exp. 3)
        ent = self.model.logger.name_to_value.get("train/entropy_loss")
        lens = [e["l"] for e in self.model.ep_info_buffer]
        mean_len = float(np.mean(lens)) if lens else 0.0
        if (self.num_timesteps >= WARMUP and ent is not None
                and ent > -0.3 and mean_len < 4000):
            self.pruned, self.reason = True, f"colapso (entropy={ent:.2f}, len={mean_len:.0f})"
            return False

        if self.trial.should_prune():
            self.pruned, self.reason = True, "peor que la mediana"
            return False
        return True


def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
        ent_coef=trial.suggest_float("ent_coef", 1e-4, 0.02, log=True),
        n_epochs=trial.suggest_int("n_epochs", 3, 10),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
    )
    print(f"\n=== Trial {trial.number}: {params}", flush=True)

    env = create_env(SEED, args.n_envs)
    model = PPO(
        "CnnPolicy",
        env,
        n_steps=128,
        batch_size=256,
        target_kl=0.02,
        seed=SEED,
        verbose=0,
        tensorboard_log=LOG_DIR,
        **params,
    )
    cb = OptunaCallback(trial)
    try:
        model.learn(total_timesteps=args.steps, callback=cb,
                    tb_log_name=f"optuna_t{trial.number:02d}")
    finally:
        env.close()

    if cb.pruned:
        print(f"    podado a los {model.num_timesteps} pasos: {cb.reason}", flush=True)
        raise optuna.TrialPruned()

    value = score(model)
    model.save(f"models/optuna/trial_{trial.number:02d}")
    print(f"    terminado: score={value:.2f}", flush=True)
    return value


if __name__ == "__main__":
    pruner = PatientPruner(
        MedianPruner(n_startup_trials=4, n_warmup_steps=WARMUP),
        patience=2,
    )
    try:
        sampler = optuna.samplers.GPSampler(seed=SEED, n_startup_trials=4)
    except TypeError:  # versiones antiguas de Optuna
        sampler = optuna.samplers.GPSampler(seed=SEED)

    study = optuna.create_study(
        study_name="ppo_pong",
        storage="sqlite:///optuna/optuna_pong.db",
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Warm start con configuraciones conocidas (solo la primera vez)
    if len(study.trials) == 0:
        # Exp. 3b: Exp. 3 (8 épocas, lr/clip fijos) + ent_coef intermedio
        study.enqueue_trial({"learning_rate": 3e-4, "ent_coef": 0.005,
                             "n_epochs": 8, "clip_range": 0.2})
        # Exp. 1 adaptado: estable y monótono, ahora sin lr -> 0
        study.enqueue_trial({"learning_rate": 2.5e-4, "ent_coef": 1e-4,
                             "n_epochs": 4, "clip_range": 0.1})
        # RL Zoo / CleanRL para Atari (referencia externa)
        study.enqueue_trial({"learning_rate": 2.5e-4, "ent_coef": 0.01,
                             "n_epochs": 4, "clip_range": 0.1})

    study.optimize(objective, n_trials=args.trials,
                   timeout=int(args.hours * 3600), gc_after_trial=True)

    print("\n" + "=" * 60)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Pruebas completas: {len(done)} | podadas: {len(pruned)}")
    if done:
        print(f"Mejor score: {study.best_value:.2f}")
        print(f"Mejores parámetros: {study.best_params}")
        print(f"Modelo: models/optuna/trial_{study.best_trial.number:02d}.zip")
    print("=" * 60)

    # Gráficas para la presentación (optuna/plots/). Si falla (p. ej. falta
    # plotly), el estudio ya está guardado: se pueden generar luego a mano.
    try:
        from optuna_plots import make_plots
        print("\nGenerando gráficas...")
        make_plots(study)
    except Exception as e:
        print(f"No se pudieron generar las gráficas: {e}")
        print("Instala plotly y ejecuta: python optuna_plots.py")
