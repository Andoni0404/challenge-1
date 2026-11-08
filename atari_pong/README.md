# Challenge Pong (RL)

Este challenge usa el entorno Atari **Pong** de la [Arcade Learning Environment (ALE)](https://ale.farama.org/) a traves de Gymnasium.

## Resultado

Modelo final: **`models/shape2_ft/best_model/best_model.zip`** (4,95M pasos), con **+17,63 ± 1,99** en modo estocástico y **+17,90 ± 2,22** en determinista (100 partidas nuevas, semilla 123, sin shaping; el máximo es +21).

Usamos 100 partidas nuevas porque el `best_model` del `EvalCallback` se elige como el mejor de muchas evaluaciones cortas: con 20 partidas su nota sale inflada (~0,8 puntos).

```bash
python play.py --mode agent                       # ver jugar al modelo final
python evaluate.py models/shape2_ft/best_model/best_model --episodes 100
```

## Estructura

```
atari_pong/
├── training.py        # entrenamiento PPO: experimentos (EXPERIMENTS) y reward shaping
├── evaluate.py        # evaluar un modelo guardado (determinista y estocástico)
├── play.py            # jugar como humano o ver al agente (por defecto, el modelo final)
├── utils.py           # helpers de play.py (código del profesor)
├── optuna_pong.py     # búsqueda de hiperparámetros con Optuna
├── optuna_plots.py    # gráficas del estudio de Optuna
├── requirements.txt
├── tools/check_ram.py # localiza en la RAM la pelota y la pala (lo usa el shaping)
├── legacy/            # versión antigua de training.py (sin uso)
├── models/<run>/      # best_model/ y final_model.zip
├── logs/
│   ├── tensorboard/   # curvas de TensorBoard (optuna/ = trials de Optuna)
│   ├── eval/<run>/    # evaluations.npz del EvalCallback
│   └── stdout/<run>.log  # salida de consola (evaluación final al final del fichero)
└── optuna/
    ├── optuna_pong.db       # estudio de Optuna (SQLite, permite retomarlo)
    ├── resumen_optuna.txt
    └── plots/               # HTML interactivos
```

Todos los scripts se ejecutan desde `atari_pong/` (las rutas son relativas).
TensorBoard: `tensorboard --logdir logs/tensorboard`.

## `training.py`

```bash
python training.py --exp <experimento> [--steps N] [--seed S] [--name RUN]
                   [--resume modelo.zip] [--lr_decay LR]
```

- `--exp`: una configuración de `EXPERIMENTS`: `baseline`, `exp1`, `exp2`, `exp3`, `exp3b`, `opt`, `opt_shape` (shaping v1) u `opt_shape2` (shaping v2).
- El **nombre del run** (carpeta en `models/`, `logs/eval/` y `logs/tensorboard/`) es el de `--exp`. Con una semilla distinta de 42 se añade `_s<seed>`, y con `--name` se puede poner uno propio. La primera línea del log indica qué `--exp` se usó.
- `--resume`: continúa entrenando un modelo con los hiperparámetros del `--exp` dado.
- `--lr_decay`: solo junto a `--resume`. Hace que el learning rate baje linealmente desde ese valor hasta 0 (afinado final).
- Para guardar la salida de consola: `nohup python training.py ... > logs/stdout/<run>.log 2>&1 &`.

**Reward shaping** (solo en entrenamiento; la evaluación usa la recompensa real). Está basado en potenciales y lee la RAM:
- v1 (`opt_shape`): premia que la pala esté alineada con la posición *actual* de la pelota.
- v2 (`opt_shape2`): predice dónde *llegará* la pelota, teniendo en cuenta los rebotes, y premia estar ahí. Da más peso a las bolas inclinadas.

## Experimentos

Los comandos están reconstruidos a partir de los logs. Las puntuaciones son las de la evaluación final (20 partidas, modo determinista).

| Run | Comando | Qué se probó | Resultado |
|---|---|---|---|
| `exp1` | `--exp exp1` | Estabilizar el baseline (8 envs, lr/clip decrecientes) | −13,45 |
| `exp2` | `--exp exp2` | exp1 + más entropía | −20,15 |
| `exp3` | `--exp exp3` | Calibrar la velocidad con `target_kl` (interrumpido) | −21 (colapso de entropía) |
| Optuna | `python optuna_pong.py --trials 15 --steps 400000 --hours 12` | 31 trials, 13 completos y 18 podados; el mejor fue el #4 | −11,99 a 400k |
| `opt` | `--exp opt --steps 3000000` | Los parámetros del trial #4 entrenados a fondo | +15,75 |
| `opt_s1` / `opt_s2` | `--exp opt --seed 1\|2 --steps 400000` | Validar con otras semillas | −9,75 / −7,40 |
| `opt_cont` | `--exp opt --resume models/opt/best_model/best_model --name opt_cont --steps 1500000` | Seguir entrenando `opt` | +16,60 |
| `opt_shape_s1` | `--exp opt_shape --seed 1 --steps 400000` | Shaping v1 | −16,30 (empeora) |
| `opt_shape_s2` | `--exp opt_shape --seed 2 --steps 400000` | Shaping v1 (interrumpido) | — |
| `opt_shape2_s1` | `--exp opt_shape2 --seed 1 --steps 400000` | Shaping v2 desde cero | −9,25 |
| `shape2_cont` | `--exp opt_shape2 --resume models/opt/best_model/best_model --name shape2_cont --steps 1500000` | `opt` + shaping v2 | +16,55 |
| **`shape2_ft`** | `--exp opt_shape2 --resume models/shape2_cont/best_model/best_model --name shape2_ft --steps 1500000 --lr_decay 2.444e-4` | Afinado: lr 2,444e-4 → 0 | **+17,85** ⭐ |

**Comparación final con 100 partidas nuevas** (`opt_cont` y `shape2_cont` parten del mismo modelo `opt`, así que `opt_cont` es el control sin shaping; `shape2_ft` continúa `shape2_cont`):

| Run | Determinista | Estocástico |
|---|---|---|
| `opt_cont` (sin shaping) | +16,30 | +16,29 |
| `shape2_cont` (shaping v2) | +17,07 | +16,50 |
| **`shape2_ft`** (shaping v2 + afinado) | **+17,90 ± 2,22** | **+17,63 ± 1,99** |

En `models/` solo se conservan `PPO/` (el baseline original), `opt/` y `shape2_ft/`.

## Documentacion del entorno

- **Pong (Gymnasium, Atari environments)**: descripcion del entorno, acciones, espacio de observacion y variantes (`Pong-v0`, `Pong-v4`, `ALE/Pong-v5`, etc.)
  https://gymnasium.farama.org/v0.28.0/environments/atari/pong/#pong

- **Entornos de la Arcade Learning Environment (ALE)**: listado completo de juegos Atari soportados, espacio de acciones comun, tipos de observacion (`rgb`, `grayscale`, `ram`), estocasticidad (sticky actions, frame skipping) y esquema de nombres de versiones (`v0`, `v4`, `v5`).
  https://ale.farama.org/v0.10.0/environments

- **Frame skipping y preprocesado en Deep Q-Networks para juegos Atari 2600**: articulo que explica en detalle como funciona el frame skipping, el redimensionado/conversion a escala de grises, el "frame stacking" y el maximo por pixel entre frames consecutivos usado por DeepMind.
  https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

## Notas clave sobre Pong

- **Acciones**: `Discrete(6)` -> `NOOP`, `FIRE`, `RIGHT`, `LEFT`, `RIGHTFIRE`, `LEFTFIRE`. En Pong, `RIGHT`/`LEFT` mueven la pala hacia arriba/abajo (son direcciones del joystick original, no de la pantalla).
- **Observacion**: por defecto `rgb` con forma `(210, 160, 3)`. Tambien existen variantes `grayscale` y `ram` (128 bytes de memoria de la consola).
- **Version recomendada**: `ALE/Pong-v5` (namespace `ALE/`), que sigue las buenas practicas descritas en Machado et al. (2018): sticky actions activas y frame skipping fijo, sin las variantes historicas `Deterministic`/`NoFrameskip`.
- **Frame skipping**: en `training.py` se usa `frameskip=1` en el entorno base y se delega el frame skipping (`frame_skip=4`) al wrapper de Atari de `stable-baselines3`, evitando aplicarlo dos veces.
- **Frame stacking**: se apilan 4 frames consecutivos (`VecFrameStack(n_stack=4)`) para que el agente pueda inferir movimiento/velocidad de la pelota, tal y como describe el articulo de preprocesado enlazado arriba.

## Uso rapido

Jugar como humano:

```bash
python play.py --mode human
```

Ver jugar a un agente entrenado (por defecto, el modelo final):

```bash
python play.py --mode agent
python play.py --mode agent --model models/opt/best_model/best_model
```

Entrenar un agente:

```bash
python training.py --exp opt
```