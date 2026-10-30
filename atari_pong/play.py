import argparse

import ale_py
import gymnasium as gym
from utils import play_human, play_agent

ENV_ID = "ALE/Pong-v5"
gym.register_envs(ale_py)



def play_random(env_id):
    """Standard random agent plays Pong against Atari's built-in opponent."""

    # env = gym.make(env_id, render_mode="rgb_array", frameskip=4)
    env = gym.make(env_id, render_mode="human", frameskip=4)

    for episode_counter in range(10):
        
        # cada nuevo episdio se reinicia el entorno
        episode_steps = 0
        episode_reward = 0
        obs = env.reset()
        
        while True:
            action = env.action_space.sample()
            obs, rewards, terminated, truncated, infos = env.step(action)
            episode_reward += rewards
            env.render()
            episode_steps += 1
            
            if terminated or truncated:
                print(f"[{episode_counter}]Episode finished. Reward: {episode_reward:.2f} (steps {episode_steps})")
                episode_reward = 0
                break
                
    env.close()
# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Play Atari Pong as a human or using a PPO agent.")
    parser.add_argument("--mode", choices=["human", "agent","random"], default="random", help="Who controls the paddle.")
    parser.add_argument("--model", type=str, default="models/PPO/best_model/best_model", help="Path to the PPO model.")
    args = parser.parse_args()

    if args.mode == "human":
        play_human(ENV_ID)
    elif args.mode == "agent":
        play_agent(args.model, ENV_ID)
    else:
        play_random(ENV_ID)


if __name__ == "__main__":
    main()
