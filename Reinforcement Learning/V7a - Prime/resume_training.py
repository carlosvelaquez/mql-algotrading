import warnings
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback
from prime import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='prime')

model = PPO2.load("prime_resume", env=make_vec_env(
    lambda: Prime(normalize=True), n_envs=128), nminibatches=32, gamma=0,
    verbose=1, tensorboard_log="./tb/", learning_rate=0.0001)

# Train the agent
model.learn(total_timesteps=int(3e7), callback=checkpoint_callback)

# Save the agent
model.save("prime_model")

print("Training done!")
