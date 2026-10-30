# Challenge Pong (RL)

Este challenge usa el entorno Atari **Pong** de la [Arcade Learning Environment (ALE)](https://ale.farama.org/) a traves de Gymnasium.

## Estructura relevante

- `training.py`: entrenamiento de un agente PPO con `stable-baselines3` usando `make_atari_env` (frame skipping + frame stacking).
- `play.py`: jugar como humano (con teclado, via `pygame`) o ver jugar a un agente PPO entrenado.
- `requirements.txt`: dependencias del challenge.

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

Ver jugar a un agente entrenado:

```bash
python play.py --mode agent --model models/PPO/best_model/best_model
```

Entrenar un agente:

```bash
python training.py
```
