import warnings
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.callbacks import CheckpointCallback
from prime import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

env = VecNormalize(make_vec_env(lambda: Prime(), n_envs=128), gamma=0)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='prime')
model = PPO2('MlpLstmPolicy', env, nminibatches=32, gamma=0,
             verbose=1, tensorboard_log="./tb/")

# Train the agent
model.learn(total_timesteps=int(3e7), callback=checkpoint_callback)

# Save the agent
model.save("prime_model")
env.save("prime_vecenv")

print("Training done!")
