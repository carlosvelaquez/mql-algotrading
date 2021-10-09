import warnings
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback
from summit import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def make_env():
    return Summit()


env = make_vec_env(make_env, n_envs=16)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='summit')
model = PPO2('MlpLnLstmPolicy', env, nminibatches=4,
             verbose=1, tensorboard_log="./tb/", gamma=0.9999)

# Train the agent
model.learn(total_timesteps=int(1e7), callback=checkpoint_callback)

# Save the agent
model.save("summit_model")

print("Training done!")
