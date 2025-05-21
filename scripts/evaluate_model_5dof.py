import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from envs.pepper_arm_env import PepperArmEnv5DOF

def evaluate(model_path: str, n_episodes: int = 100):
    model = PPO.load(model_path)
    env = PepperArmEnv5DOF(render_mode='human')
    successes = 0
    distances = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        dist = info['distance']
        distances.append(dist)
        if terminated:
            successes += 1
    print(f"Success rate: {successes/n_episodes*100:.2f}%")
    print(f"Average final distance: {np.mean(distances):.4f} m")

if __name__=='__main__':
    evaluate('models/ppo_pepper5dof.zip')