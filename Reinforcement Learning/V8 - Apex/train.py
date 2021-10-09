import warnings
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.callbacks import CheckpointCallback
from apex import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def make_env():
    return Apex(
        max_account_loss=.05, min_target_points=25, max_target_points=250, max_opt_bars=60, min_risk_ratio=.5)


env = VecNormalize(make_vec_env(make_env, n_envs=16),
                   clip_obs=1.0, clip_reward=1.0, gamma=0)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='apex')
model = PPO2('MlpLnLstmPolicy', env, nminibatches=4, gamma=0,
             verbose=1, tensorboard_log="./tb/")

# Train the agent
model.learn(total_timesteps=int(1e7), callback=checkpoint_callback)

# Save the agent
model.save("apex_model")
env.save("apex_vecenv")

print("Training done!")
