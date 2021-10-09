import warnings
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.callbacks import CheckpointCallback
from stellar import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def make_env():
    return Stellar(history_path="./history", disable_done=True)


env = make_vec_env(make_env, n_envs=128)

model = PPO2.load("stellar_model")
model.set_env(env)

obs = env.reset()
total_reward = 0
rewards_number = 0
total_trade_reward = 0
trade_rewards_number = 0
target_i = 1000

state = None
done = [False for _ in range(env.num_envs)]

for i in range(target_i):
    action, state = model.predict(obs, state=state, mask=done)
    obs, rewards, dones, info = env.step(action)

    print("Evaluating... {}/{}".format(i, target_i))

    for x in rewards:
        total_reward += x
        rewards_number += 1

print("Mean: {}".format(total_reward/rewards_number))
