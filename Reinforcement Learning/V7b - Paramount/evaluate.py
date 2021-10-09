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

env = VecNormalize.load(
    "paramount_vecenv", make_vec_env(lambda: Paramount(debug_mode=0), n_envs=128))
env.gamma = 0.999
env.training = False

model = PPO2.load("paramount_model", env=env, nminibatches=32, gamma=0.999,
                  verbose=1, tensorboard_log="./tb/")

obs = env.reset()
state = None
done = [False for _ in range(env.num_envs)]

days = 1
target_i = 1000000

for i in range(target_i):
    # We need to pass the previous state and a mask for recurrent policies
    # to reset lstm state when a new episode begin
    action, state = model.predict(obs, state=state, mask=done)
    obs, reward, done, info = env.step(action)

    n = reward.shape[0]

    total_trade_reward = 0
    total_target_points = 0
    total_risk = 0
    total_trades = 0

    for j in range(n):
        if done[j]:
            days += 1

        total_trade_reward += info[j]["total_trade_reward"]
        total_trades += info[j]["total_trades"]
        total_risk += info[j]["total_risk"]
        total_target_points += info[j]["total_target_points"]

    mean_reward = total_trade_reward/total_trades
    mean_r_day = total_trade_reward/days
    mean_target = total_target_points/total_trades
    mean_risk = total_risk/total_trades

    print("[Evaluating {}/{}] Total Trades: {:d}, Mean Reward: {:.5f}, Days: {:d}, Mean R/day: {:.2f}, Mean Target Pts: {:.2f}, Mean Risk: {:.2f}".format(
        i, target_i, total_trades, mean_reward, days, mean_r_day, mean_target, mean_risk), end='\r')
