from stable_baselines import PPO2, A2C
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from hellbent import *


def make_env():
    return Hellbent(history_path="./history")


env = make_vec_env(make_env, n_envs=16)

env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=1.)

model = PPO2('MlpLstmPolicy', env, nminibatches=1, verbose=1)
#model = A2C('MlpLstmPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=int(25000000))
# Save the agent
model.save("hellbent_model")
env.save("hellbent_vecnormalize")

print("done")
