import sys
import gymnasium as gym

# for agent mode
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# ============================================================
# HUMAN MODE
# ============================================================
def play_human(env_id):
    """Human plays Pong against Atari's built-in opponent.
    Controls: UP/W move up, DOWN/S move down, SPACE fire, ESC quit.
    """
    try:
        import pygame
    except ImportError:
        print("Pygame is required for human mode.\nInstall it with:\n\n    pip install pygame")
        sys.exit(1)

    # frameskip=4 means every keyboard action is repeated for ~4 Atari frames.
    env = gym.make(env_id, render_mode="rgb_array", frameskip=4)
    obs, info = env.reset()

    action_meanings = env.unwrapped.get_action_meanings()
    print("Available actions:")
    for i, action in enumerate(action_meanings):
        print(f"  {i}: {action}")

    NOOP = action_meanings.index("NOOP")
    FIRE = action_meanings.index("FIRE")
    # Atari Pong uses RIGHT/LEFT joystick directions even though the paddle moves vertically.
    MOVE_UP = action_meanings.index("RIGHT")
    MOVE_DOWN = action_meanings.index("LEFT")

    pygame.init()
    frame = env.render()
    height, width, _ = frame.shape
    SCALE = 3
    screen = pygame.display.set_mode((width * SCALE, height * SCALE))
    pygame.display.set_caption("Pong - Human vs Atari")
    clock = pygame.time.Clock()
    total_reward = 0
    running = True

    print("\n" + "=" * 40)
    print("HUMAN vs ATARI")
    print("=" * 40)
    print("UP / W   : move up")
    print("DOWN / S : move down")
    print("SPACE    : FIRE")
    print("ESC      : quit")
    print("=" * 40 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = MOVE_UP
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = MOVE_DOWN
        elif keys[pygame.K_SPACE]:
            action = FIRE
        else:
            action = NOOP

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = env.render()
        # Gymnasium image is height x width x RGB, pygame expects width x height x RGB
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        frame_surface = pygame.transform.scale(frame_surface, (width * SCALE, height * SCALE))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        # Atari runs at ~60 frames/s; each action covers 4 frames -> 15 decisions/s.
        clock.tick(15)

        if terminated or truncated:
            print(f"\nGame finished. Reward: {total_reward}")
            total_reward = 0
            obs, info = env.reset()

    env.close()
    pygame.quit()
    
# ============================================================
# AGENT MODE
# ============================================================
def play_agent(model_path, env_id):
    """Watch a trained PPO agent play against Atari."""
    print(f"Loading model: {model_path}")

    # IMPORTANT: this preprocessing must be identical to training.py.
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=123,
        env_kwargs={"frameskip": 1, "render_mode": "human"},
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "terminal_on_life_loss": False,
            "clip_reward": False,
        },
    )
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(model_path, env=env)
    obs = env.reset()
    episode_reward = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            episode_reward += rewards[0]
            env.render()

            if dones[0]:
                print(f"Episode finished. Reward: {episode_reward:.2f}")
                episode_reward = 0
                # VecEnv automatically resets after done, no manual reset needed.
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()