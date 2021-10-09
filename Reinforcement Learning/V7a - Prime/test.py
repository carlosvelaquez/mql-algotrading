import gym
from prime import *
import random

env = Prime()
env.reset()

for _ in range(250):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    print("\nAction: {}, Reward: {}, Observation Shape: {}, Obs:".format(
        action, reward, obs.shape))
    print(obs)
    print("Done: {}\n".format(done))

    if done:
        env.reset()
