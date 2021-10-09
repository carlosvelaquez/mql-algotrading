from stable_baselines import PPO2, A2C
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from prodigy import *


def make_env():
    return Prodigy(mt5_path=None, single_step=True,
                   csv_path="./prod_history.csv", debug_mode=1)


env = make_vec_env(make_env, n_envs=4)

env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=1.)

#model = PPO2('MlpLnLstmPolicy', env, nminibatches=1, verbose=1)
model = A2C('MlpLnLstmPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=int(25000000))
# Save the agent
model.save("prodigy_model")

print("done")
