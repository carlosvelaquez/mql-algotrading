import warnings
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.callbacks import CheckpointCallback
from zenith import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

env = VecNormalize(make_vec_env(lambda: Zenith(), n_envs=128))
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='zenith')
model = PPO2('MlpPolicy', env, nminibatches=32,
             verbose=1, tensorboard_log="./tb/")

# Train the agent
model.learn(total_timesteps=int(3e7), callback=checkpoint_callback)

# Save the agent
model.save("zenith_model")
env.save("zenith_vecenv")

print("Training done!")
