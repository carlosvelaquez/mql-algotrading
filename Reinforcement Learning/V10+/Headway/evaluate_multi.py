import warnings
import matplotlib.pyplot as plt
import numpy as np
import os

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from headway import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

target_days = 150
max_loss = 250000
secured_percentage = .08
balance = 0


def make_env():
    return Headway(gamma=1, warmup_steps=0, max_loss=max_loss, single_position=False, trailing_end=False, debug=0)


env = make_vec_env(make_env, n_envs=16)
model = PPO2.load("headway_model")

obs = env.reset()
domain = obs.shape[0]
state = None
done = [False for _ in range(env.num_envs)]

total_reward = 0
total_diff = 0
total_secured = 0
min_pl = 0
min_balance = balance
total_orders = 0
total_bars = 0

days = 0
steps = 0

hold_action = np.array([[0, 0] for _ in range(env.num_envs)])


# fig, (prob_axis, target_axis) = plt.subplots(2)

# prob_axis.set(xlabel="Probability", ylabel="P/L")
# target_axis.set(xlabel="Target", ylabel="P/L")
# prob_axis.grid(True)
# target_axis.grid(True)

plt.grid(True)
# plt.xlabel("Days")
# plt.ylabel("USD")
# plt.title("Profit over time")

daysPlot = []
balancePlot = []
securedPlot = []

files = []

for i in range(domain):
    f = files.append(
        open(os.path.join("./studies", "study_{:d}.csv".format(i)), "w"))


while days < target_days:
    steps += domain

    action, state = model.predict(
        obs, state=state, mask=done, deterministic=True)

    prob = model.action_probability(
        obs, state=state, mask=done, actions=action).flatten()

    obs, reward, done, info = env.step(action)

    diffs = []

    for x in info:
        diffs.append(x["diff"])

    # prob_axis.plot(prob, diffs, 'go', alpha=.1)
    # target_axis.plot(action[:, 0], diffs, 'bo', alpha=.1)

    actives = 0
    warms = 0

    for j in range(domain):
        rw = info[j]["diff"]

        if info[j]["warming_up"]:
            warms += 1

            if rw != 0:
                out = ""

                for e in obs[j]:
                    out += str(e)
                    out += ","

                out += str(rw)
                out += str("\n")

                if done[j]:
                    out += "\n"

                files[j].write(out)

        if info[j]["active_order"]:
            actives += 1

        if info[j]["sequence_changed"]:
            seq = info[j]["last_sequence"]
            plt.clf()

            plt.plot(np.arange(0, seq.shape[0]), seq[:, 1])

        if rw != 0:
            total_diff += rw
            total_orders += 1
            min_pl = min(total_diff, min_pl)

            # pos_size = max((balance/(max_loss*2))/(.00001*max_loss), 1000)
            # pos_size = min(pos_size, 200*100000)
            pos_size = 1000

            profit = rw*pos_size

            if profit > 0:
                invested = profit*(1-secured_percentage)
                secured = profit*secured_percentage
                # invested = 0
                # secured = 0
            else:
                invested = profit
                secured = 0

            balance += invested
            total_secured += secured
            min_balance = min(balance, min_balance)

        if done[j]:
            days += 1
        #     # total_diff += info[j]["total_diff"]
        #     # total_reward += info[j]["total_reward"]
        #     # total_orders += info[j]["total_orders"]
        #     # total_bars += info[j]["total_bars"]

        #     min_pl = min(total_diff, min_pl)

        #     pos_size = max((balance/200)/(.00001*max_loss), 1000)
        #     pos_size = min(pos_size, 200*100000)

        #     profit = (info[j]["total_diff"])*pos_size

        #     if profit > 0:
        #         invested = profit*(1-secured_percentage)
        #         secured = profit*secured_percentage
        #     else:
        #         invested = profit
        #         secured = 0

        #     balance += invested
        #     total_secured += secured
        #     min_balance = min(balance, min_balance)

        #     if total_orders == 0:
        #         total_orders = 1

        #     print(
        #         "Day {:d}/{:d}, Rw: {:.5f}, Pts: {:.5f}, Min Pts: {:.5f}, Orders: {:d}, Mean Bars: {:.2f}, Balance: {:.2f}, Min Balance: {:.2f}, Secured: {:.2f}".format(days, target_days, total_reward, total_diff, min_pl, total_orders, total_bars/total_orders, balance, min_balance, total_secured), end='\r')

        #     daysPlot.append(days)
        #     balancePlot.append(balance)
        #     securedPlot.append(total_secured)

        # print(
        #     "Day {:d}/{:d}, Pts: {:.5f}, Min Pts: {:.5f}, Orders: {:d}, Mean Bars: {:.2f}, Balance: {:.2f}, Min Balance: {:.2f}, Secured: {:.2f}".format(days, target_days, total_reward, total_diff, min_pl, total_orders, total_bars/total_orders, balance, min_balance, total_secured), end='\r')

    print(
        "Day {:d}/{:d}, Steps: {:d}, Warm Envs: {:d}, Active Orders: {:d}, Pts: {:.5f}, Min Pts: {:.5f}, Orders: {:d}, Balance: {:.2f}, Min Balance: {:.2f}, Secured: {:.2f}".format(days, target_days, steps, warms, actives, total_diff, min_pl, total_orders, balance, min_balance, total_secured), end='\r')


plt.plot(daysPlot, balancePlot, 'b-', label="Balance")
plt.plot(daysPlot, securedPlot, 'g-', label="Secured")


plt.legend()
plt.show()
