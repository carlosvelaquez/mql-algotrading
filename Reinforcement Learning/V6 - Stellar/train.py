import warnings
from stable_baselines import PPO2, A2C
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.callbacks import CheckpointCallback
from stellar import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def make_env():
    return Stellar(history_path="./history")


env = make_vec_env(make_env, n_envs=128)

# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=1.)
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./checkpoints/',
                                         name_prefix='stellar')
model = PPO2('MlpLstmPolicy', env, nminibatches=32,
             n_steps=2048, gamma=0, verbose=1, learning_rate=0.0001, tensorboard_log="./tb/")
#model = A2C('MlpLstmPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=int(25000000))
# Save the agent
model.save("stellar_model")

print("done")
