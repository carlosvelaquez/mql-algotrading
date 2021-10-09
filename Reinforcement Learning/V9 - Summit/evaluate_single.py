import warnings
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from summit import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def make_env():
    return Summit(
        max_account_loss=.05, min_target_points=25, max_target_points=250, max_opt_bars=60, min_risk_ratio=.5, debug_mode=0)


env = make_env()

vec_env = VecNormalize.load(
    "summit_vecenv", make_vec_env(make_env, n_envs=16))

vec_env.training = False
vec_env.norm_reward = False

model = PPO2.load("summit_model")
num_envs = vec_env.num_envs

obs = env.reset()
state = None
done = False

total_reward = 0
reward_count = 0

target_days = 1000
days = 0
total_pl = 0
real_pl = 0
steps = 0

hold_action = np.array([[0, 0] for _ in range(num_envs)])

fig, axes = plt.subplots(2, 2)

axes[0, 0].set(xlabel="Probability", ylabel="Reward")
axes[0, 1].set(xlabel="Collapsed Probability", ylabel="Reward")
axes[1, 0].set(xlabel="Standard Deviation", ylabel="Reward")
axes[1, 1].set(xlabel="Collapsed Standard Deviation", ylabel="Reward")

axes[0, 0].grid(True)
axes[0, 1].grid(True)
axes[1, 0].grid(True)
axes[1, 1].grid(True)

initial_balance = env.initial_balance

while days < target_days:
    steps += 1
    mask = [done for _ in range(num_envs)]

    # We need to pass the previous state and a mask for recurrent policies
    # to reset lstm state when a new episode begin
    norm_obs = vec_env.normalize_obs(obs)
    multi_obs = [norm_obs for _ in range(num_envs)]

    action, state = model.predict(
        multi_obs, state=state, mask=mask, deterministic=True)

    # prob = model.action_probability(
    #     multi_obs, state=state, mask=mask, actions=action).flatten()

    # std_dev = model.action_probability(
    #     multi_obs, state=state, mask=mask)[0][:, 1]

    # for i in range(domain):
    #     if std_dev[i] < -3.1 or std_dev[i] > -3:
    #         action[i][0] = 0
    #         action[i][1] = 0

    obs, reward, done, info = env.step(action[0])

    if done:
        days += 1

        total_pl += info["total_pl"]
        real_pl += info["real_pl"]

        if days > 0:
            pf = (initial_balance + total_pl)/initial_balance
            rpf = (initial_balance + real_pl)/initial_balance
            mean_pl = total_pl/days
            mean_rpl = real_pl/days
            dpf = (initial_balance + mean_pl)/initial_balance
            drpf = (initial_balance + mean_rpl)/initial_balance

            print("[Evaluating... Day {}/{}] S: {:d}, P/L: {:.2f}, M P/L: {:.2f}, PF: {:.2f}, DPF: {:.2f}, R-P/L: {:.2f}, M R-P/L: {:.2f}, RPF: {:.2f}, DRPF: {:.2f}".format(
                days, target_days, steps, total_pl, mean_pl, pf, dpf, real_pl, mean_rpl, rpf, drpf), end='\r')

        env.reset()

# plt.show()
