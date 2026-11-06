import numpy as np
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", frameskip=4, repeat_action_probability=0.0)
env.reset(seed=0)
names = env.unwrapped.get_action_meanings()
UP, DOWN, NOOP = names.index("RIGHT"), names.index("LEFT"), names.index("NOOP")

rams = []
for a in [UP] * 15 + [DOWN] * 15:
    env.step(a)
    rams.append(env.unwrapped.ale.getRAM().copy())
rams = np.array(rams, dtype=int)
up_delta = rams[14] - rams[0]
down_delta = rams[29] - rams[15]
cands = [i for i in range(128) if up_delta[i] * down_delta[i] < 0 and abs(up_delta[i]) > 5]
print("Bytes que se mueven con la pala propia:", cands)
for i in cands:
    print(f"  ram[{i}]: subiendo {rams[0][i]}->{rams[14][i]} | bajando {rams[15][i]}->{rams[29][i]}")

env.reset(seed=1)
print("\nPaso | ram49 | ram54 | ram50 | ram51 | reward")
for t in range(90):
    _, r, term, trunc, _ = env.step(NOOP)
    ram = env.unwrapped.ale.getRAM()
    if t % 3 == 0:
        print(f"{t:4d} | {ram[49]:5d} | {ram[54]:5d} | {ram[50]:5d} | {ram[51]:5d} | {r}")
    if term or trunc:
        break
env.close()