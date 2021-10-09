import warnings
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.callbacks import CheckpointCallback
from paramount import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

env = VecNormalize(make_vec_env(lambda: Paramount(), n_envs=128), gamma=.999)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='')
model = PPO2('MlpLstmPolicy', env, nminibatches=32, gamma=.999,
             verbose=1, tensorboard_log="./tb/")

# Train the agent
model.learn(total_timesteps=int(5e7), callback=checkpoint_callback)

# Save the agent
model.save("paramount_model")
env.save("paramount_vecenv")

print("Training done!")
