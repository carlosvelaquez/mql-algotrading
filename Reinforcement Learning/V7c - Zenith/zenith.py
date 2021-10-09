import os
import gym
import random
import numpy as np
import pandas as pd
from gym import spaces

NUM_FEATURES = 20


class Apex(gym.Env):
    def __init__(self, history_path="./history", initial_balance=200, lot_size=100000, commission=3.5, obs_timesteps=1500, min_target_points=5, max_target_points=100, min_risk_ratio=1, max_risk_ratio=3, account_risk=0.01, max_account_loss=.1, gamma=.8, debug_mode=1):
        # Parameters
        self.history_path = history_path
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.commission = commission
        self.obs_timesteps = obs_timesteps
        self.debug_mode = debug_mode
        self.min_target_points = min_target_points
        self.max_target_points = max_target_points
        self.min_risk_ratio = min_risk_ratio
        self.max_risk_ratio = max_risk_ratio
        self.account_risk = account_risk
        self.max_account_loss = max_account_loss
        self.gamma = gamma
        self.debug_mode = debug_mode

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_timesteps, NUM_FEATURES))

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]))

        # Load history info
        file = open(os.path.join(history_path, "meta.txt"), "r")
        self.history_file_count = int(file.read())

        self.log("Init", "Found {} history files".format(
            self.history_file_count), 1)

    def load_random_history_file(self):
        index = random.randint(0, self.history_file_count)

        df = pd.read_csv(os.path.join(
            self.history_path, "{}.csv".format(index)))
        df = df.drop(columns=["Bid", "Ask"])

        self.history = df.to_numpy()
        self.history_size = self.history.shape[0]
        self.current_history_file = index

        self.current_pos = self.obs_timesteps

    def get_observation(self):
        start_index = self.current_pos - self.obs_timesteps
        end_index = self.current_pos
        obs = np.column_stack(
            (self.history[start_index:end_index], self.balance_history))

        self.current_price = self.history[self.current_pos][3]
        self.current_pos += 1

        if (self.current_pos >= self.history_size - 1) or (self.balance <= self.min_balance):
            self.done = True

        balance_calc = (1 - self.max_account_loss)*self.balance

        if self.min_balance < balance_calc:
            self.min_balance = balance_calc

        if self.done:
            self.log("Episode End", "Reward: {:.2f} | Final Balance {:.2f} | P/L: {:.2f} | Orders: {:d} | Mean Order P/L: {:.2f} | Mean Bars/Order {:.2f}".format(
                self.total_reward, self.balance, self.total_pl, self.total_orders, self.total_pl /
                self.total_orders, self.total_bars/self.total_orders
            ), 1)

        return obs

    def reset(self):
        # Init trackers
        self.done = False
        self.balance = self.initial_balance
        self.balance_history = np.full(
            shape=(self.obs_timesteps,), fill_value=self.balance)
        self.min_balance = (1 - self.max_account_loss)*self.balance
        self.total_reward = 0
        self.total_pl = 0
        self.total_bars = 0
        self.total_orders = 0

        self.load_random_history_file()
        return self.get_observation()

    def advance_balance(self, new_balance):
        self.balance_history = np.delete(self.balance_history, 0)
        self.balance_history = np.append(
            self.balance_history, new_balance)

    def advance_order(self, order):
        bars_held = 0

        tp = order["tp"]
        sl = order["sl"]
        open_price = order["open_price"]
        pos_size = order["pos_size"]

        long_order = False
        short_order = False

        if tp > sl:
            upper_bound = tp
            lower_bound = sl
            long_order = True
        elif sl > tp:
            upper_bound = sl
            lower_bound = tp
            short_order = True

        prev_open = self.history[self.current_pos][0]

        self.balance -= self.commission*(pos_size/self.lot_size)
        total_balance = 0

        finished = False

        for i in range(self.current_pos, self.history_size - 1):
            # Check for new bar
            if prev_open != self.history[i][0]:
                bars_held += 1
                prev_open = self.history[i][0]

            ref_price = self.history[i][3]

            if long_order:
                price_diff = ref_price - open_price
            elif short_order:
                price_diff = open_price - ref_price

            floating_pl = price_diff*pos_size

            if ref_price >= upper_bound or ref_price <= lower_bound:
                self.balance += floating_pl - \
                    self.commission*(pos_size/self.lot_size)

                total_balance += self.balance*pow(self.gamma, bars_held)

                self.advance_balance(self.balance)

                self.total_pl += floating_pl
                self.total_bars += bars_held
                self.current_pos = i
                finished = True
                break

            total_balance += (self.balance + floating_pl) * \
                pow(self.gamma, bars_held)
            self.advance_balance(self.balance + floating_pl)

        if not finished:
            self.current_pos = self.history_size - 2

        return total_balance

    def step(self, action):
        target_percentage = action[0]
        risk_percentage = action[1]

        reward = self.balance

        if target_percentage != 0:
            self.total_orders += 1

            if target_percentage > 0:
                base_points = self.min_target_points
            elif target_percentage < 0:
                base_points = -self.min_target_points

            risk = self.min_risk_ratio + \
                ((self.max_risk_ratio - self.min_risk_ratio)*risk_percentage)

            target_points = base_points + \
                (target_percentage*(self.max_target_points-self.min_target_points))
            risk_points = target_points*risk

            open_price = self.current_price
            pos_size = (self.balance*self.account_risk) / \
                (abs(risk_points)*.00001)

            order = {
                "open_price": open_price,
                "tp": round(open_price + (target_points*.00001), 5),
                "sl": round(open_price - (risk_points*.00001), 5),
                "pos_size": pos_size
            }

            reward = self.advance_order(order)

        self.total_reward += reward
        return self.get_observation(), reward, self.done, {}

    def log(self, prefix, msg, debug_level=2):
        if debug_level <= self.debug_mode:
            print("[{}] {}".format(prefix, msg))

    def render(self, mode='human'):
        pass
