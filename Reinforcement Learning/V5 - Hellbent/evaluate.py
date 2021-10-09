from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.evaluation import evaluate_policy

from hellbent import *


def make_env():
    return Hellbent(history_path="./history", debug_mode=0)


env = VecNormalize.load("hellbent_vecnormalize",
                        make_vec_env(make_env, n_envs=16))

model = PPO2.load("hellbent_model")
model.set_env(env)

for _ in range(100):
    total_reward = 0
    rewards_number = 0

    obs = env.reset()

    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        for x in rewards:
            total_reward += x
            rewards_number += 1

        if True in dones:
            break

    print("Mean Reward: {:.5f}".format(total_reward/rewards_number))
