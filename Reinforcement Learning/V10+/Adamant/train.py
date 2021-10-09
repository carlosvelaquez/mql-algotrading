import warnings
from stable_baselines import A2C
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import EvalCallback
from adamant import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def make_env():
    return Adamant()


env = make_vec_env(make_env, n_envs=8)
eval_env = Adamant(single_position=True, max_loss=1000,
                   gamma=1, trailing_end=True)

eval_callback = EvalCallback(eval_env, best_model_save_path='./best/',
                             log_path='./logs/', eval_freq=10000,
                             deterministic=True, render=False)

model = A2C('MlpPolicy', env, gamma=0,
            verbose=1, tensorboard_log="./tb/")

# Train the agent
model.learn(total_timesteps=int(15e6), callback=eval_callback)

# Save the agent
model.save("adamant_model")
print("Training done!")
