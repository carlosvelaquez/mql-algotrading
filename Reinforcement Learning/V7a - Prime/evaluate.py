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

env = VecNormalize.load(
    "prime_vecenv", make_vec_env(lambda: Prime(debug_mode=0), n_envs=128))
env.gamma = 0
env.training = False

model = PPO2.load("prime_model", env=env, nminibatches=32, gamma=0,
                  verbose=1, tensorboard_log="./tb/")

obs = env.reset()
state = None
done = [False for _ in range(env.num_envs)]

total_reward = 0
n_reward = 0

total_trade_reward = 0
n_trade_reward = 0

target_i = 1000000

days = 0

for i in range(target_i):
    # We need to pass the previous state and a mask for recurrent policies
    # to reset lstm state when a new episode begin
    action, state = model.predict(obs, state=state, mask=done)
    obs, reward, done, info = env.step(action)

    for j in range(reward.shape[0]):
        if info[j]["bars_held"] != -1:
            total_reward += reward[j]
            n_reward += 1

            total_trade_reward += (info[j]["trade_reward"])*100000
            n_trade_reward += 1

    for d in done:
        if d:
            days += 1

    print("[Evaluating {}/{}] Mean R: {:.5f}, Total TR: {:.2f}, Total Ts: {:d}, Mean TR: {:.2f}, Days: {:d}, Mean TR/day: {:.2f}".format(
        i, target_i, total_reward/n_reward, total_trade_reward, n_trade_reward, total_trade_reward/n_trade_reward, days, total_trade_reward/days), end='\r')
