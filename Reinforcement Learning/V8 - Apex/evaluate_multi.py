import warnings
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from apex import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def make_env():
    return Apex(
        max_account_loss=.05, min_target_points=25, max_target_points=250, max_opt_bars=60, min_risk_ratio=.5, debug_mode=0)


apex_env = make_env()
env = VecNormalize.load(
    "apex_vecenv", make_vec_env(make_env, n_envs=16))

env.training = False
env.norm_reward = False

model = PPO2.load("apex_model", env=env)

# obs = env.reset()
obs = env.reset()
domain = obs.shape[0]
state = None
done = [False for _ in range(env.num_envs)]

total_reward = 0
reward_count = 0

target_days = 1000
days = 0
total_pl = 0
real_pl = 0
steps = 0

hold_action = np.array([[0, 0] for _ in range(env.num_envs)])

fig, axes = plt.subplots(2, 2)

axes[0, 0].set(xlabel="Probability", ylabel="Reward")
axes[0, 1].set(xlabel="Collapsed Probability", ylabel="Reward")
axes[1, 0].set(xlabel="Standard Deviation", ylabel="Reward")
axes[1, 1].set(xlabel="Collapsed Standard Deviation", ylabel="Reward")

axes[0, 0].grid(True)
axes[0, 1].grid(True)
axes[1, 0].grid(True)
axes[1, 1].grid(True)

initial_balance = apex_env.initial_balance

while days < target_days:
    steps += 1

    # We need to pass the previous state and a mask for recurrent policies
    # to reset lstm state when a new episode begin
    action, state = model.predict(
        obs, state=state, mask=done, deterministic=True)

    prob = model.action_probability(
        obs, state=state, mask=done, actions=action).flatten()

    std_dev = model.action_probability(
        obs, state=state, mask=done)[0][:, 1]

    # for i in range(domain):
    #     if std_dev[i] < -3.1 or std_dev[i] > -3:
    #         action[i][0] = 0
    #         action[i][1] = 0

    obs, reward, done, info = env.step(action)

    prob_collapsed = np.array([])
    stddev_collapsed = np.array([])

    for i in range(domain):
        # a = action[i][0]
        p = round(prob[i], 3)
        sd = round(std_dev[i], 2)
        r = reward[i]

        prob_collapsed = np.append(prob_collapsed, p)
        stddev_collapsed = np.append(stddev_collapsed, sd)

    # probs = np.unique(prob_collapsed)
    # probs = np.vstack([probs, np.zeros(shape=(probs.shape[0],))])

    # for i in range(prob_collapsed.shape[0]):
    #     for j in range(probs.shape[0]):
    #         if prob_collapsed[i] == probs[j][0]:
    #             probs[j][1] += reward[i]

    axes[0, 0].plot(prob, reward, 'go', alpha=.1)
    axes[1, 0].plot(std_dev, reward, 'bo', alpha=.1)
    axes[0, 1].plot(prob_collapsed, reward, 'go', alpha=.1)
    axes[1, 1].plot(stddev_collapsed, reward, 'bo', alpha=.1)

    for j in range(domain):
        if done[j]:
            days += 1
            inf = info[j]

            total_pl += inf["total_pl"]
            real_pl += inf["real_pl"]

    if days > 0:
        pf = (initial_balance + total_pl)/initial_balance
        rpf = (initial_balance + real_pl)/initial_balance
        mean_pl = total_pl/days
        mean_rpl = real_pl/days
        dpf = (initial_balance + mean_pl)/initial_balance
        drpf = (initial_balance + mean_rpl)/initial_balance

        print("[Evaluating... Day {}/{}] S: {:d}, P/L: {:.2f}, M P/L: {:.2f}, PF: {:.2f}, DPF: {:.2f}, R-P/L: {:.2f}, M R-P/L: {:.2f}, RPF: {:.2f}, DRPF: {:.2f}".format(
            days, target_days, steps, total_pl, mean_pl, pf, dpf, real_pl, mean_rpl, rpf, drpf), end='\r')

plt.show()
