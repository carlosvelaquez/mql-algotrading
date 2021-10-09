import os
import gym
import math
import random
import numpy as np
import pandas as pd

NUM_FEATURES = 19


class Apex(gym.Env):
    def __init__(self, history_path="./history", tensorboard_logdir="./tb", initial_balance=200, pos_size=1000, lot_size=100000, commission=3.5, min_target_points=5, max_target_points=100, min_risk_ratio=1, max_risk_ratio=3, real_account_risk=0.01, max_account_loss=.1, max_opt_bars=15, fixed_index=None, debug_mode=1):
        # Parameters
        self.history_path = history_path
        self.initial_balance = initial_balance
        self.pos_size = pos_size
        self.lot_size = lot_size
        self.commission = commission
        self.min_target_points = min_target_points
        self.max_target_points = max_target_points
        self.min_risk_ratio = min_risk_ratio
        self.max_risk_ratio = max_risk_ratio
        self.real_account_risk = real_account_risk
        self.max_account_loss = max_account_loss
        self.max_opt_bars = max_opt_bars
        self.fixed_index = fixed_index
        self.debug_mode = debug_mode

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(NUM_FEATURES,))

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]))

        # Load history info
        file = open(os.path.join(history_path, "meta.txt"), "r")
        self.history_file_count = int(file.read())

        self.log("Initializing", "Found {} history files".format(
            self.history_file_count), 1)

    def load_random_history_file(self):
        if self.fixed_index:
            index = self.fixed_index
        else:
            index = random.randint(0, self.history_file_count)

        df = pd.read_csv(os.path.join(
            self.history_path, "{}.csv".format(index)))
        df = df.drop(columns=["Bid", "Ask"])

        self.history = df.to_numpy()
        self.history_size = self.history.shape[0]
        self.current_history_file = index

    def get_observation(self):
        obs = self.history[self.current_pos]
        self.current_price = obs[3]
        self.current_pos += 1

        balance_calc = (1 - self.max_account_loss)*self.balance

        if self.min_balance < balance_calc:
            self.min_balance = balance_calc

        if self.balance > self.max_balance:
            self.max_balance = self.balance

        if self.real_balance > self.max_real_balance:
            self.max_real_balance = self.real_balance

        return obs

    def reset(self):
        # Init trackers
        self.current_pos = 0
        self.real_pos = 0
        self.done = False
        self.balance = self.initial_balance
        self.real_balance = self.initial_balance
        self.max_balance = self.balance
        self.max_real_balance = self.balance
        self.min_balance = 0
        self.total_reward = 0
        self.total_pl = 0
        self.real_pl = 0
        self.total_bars = 0
        self.total_risk = 0
        self.total_target = 0
        self.total_orders = 0
        self.short_orders = 0
        self.long_orders = 0
        self.real_orders = 0
        self.end_info = {}

        self.load_random_history_file()
        return self.get_observation()

    def advance_order(self, open_price, tp, sl, risk):
        if tp > sl:
            upper_bound = tp
            lower_bound = sl
            def get_price_diff(ref_price): return ref_price - open_price
        elif sl > tp:
            upper_bound = sl
            lower_bound = tp
            def get_price_diff(ref_price): return open_price - ref_price

        prev_open = self.history[self.current_pos][0]
        total_pl = 0
        bars_held = 1
        tick_count = 1
        floating_pl = 0

        price_diff = 0
        commissions = (self.commission*(self.pos_size/self.lot_size))*2

        for i in range(self.current_pos, self.history_size - 1):
            if prev_open != self.history[i][0]:
                bars_held += 1
                prev_open = self.history[i][0]

            ref_price = self.history[i][3]

            price_diff = get_price_diff(ref_price)
            floating_pl = (price_diff*self.pos_size) - commissions
            total_pl += floating_pl

            if ref_price <= lower_bound or ref_price >= upper_bound:
                break

            tick_count += 1

        if self.current_pos >= self.real_pos:
            real_pos_size = (
                self.balance*self.real_account_risk)/abs(open_price-sl)
            real_pl = (price_diff*real_pos_size) - commissions

            self.real_pos = self.current_pos + tick_count
            self.real_balance += real_pl
            self.real_pl += real_pl
            self.real_orders += 1

        self.balance += floating_pl
        self.total_pl += floating_pl
        self.total_bars += bars_held
        average_pl = total_pl/tick_count

        gain = floating_pl + average_pl

        if gain >= 0:
            hold_modifier = max(1 - (bars_held/self.max_opt_bars), .0001)
            return (gain/risk)*hold_modifier
        else:
            hold_modifier = min(bars_held/self.max_opt_bars, 1)
            return gain*hold_modifier

    def step(self, action):
        target_percentage = action[0]
        risk_percentage = action[1]

        reward = 0

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
            tp = round(open_price + (target_points*.00001), 5)
            sl = round(open_price - (risk_points*.00001), 5)

            reward = self.advance_order(open_price, tp, sl, risk)
            self.total_target += abs(target_points)
            self.total_reward += reward
            self.total_risk += risk

        return self.get_observation(), reward, self.check_done(), self.end_info

    def check_done(self):
        done_conditions = []

        done_conditions.append(self.current_pos >= self.history_size - 1)
        done_conditions.append(self.balance <= self.min_balance)

        if True in done_conditions:
            self.gen_info()

            if self.debug_mode >= 1:
                print("-----\nReward: {:2f}, Orders: {:d}, Mean Reward: {:.2f}, Profit Factor: {:.2f}, Real Profit Factor: {:.2f}\nMean Target: {:.2f}, Mean Risk: {:.2f}, Mean Bars: {:.2f}\nBalance: {:.2f}, Max Balance: {:.2f}, P/L: {:.2f}, Mean P/L: {:.2f}\nReal Orders: {:d}, Real Balance: {:.2f}, Max Real Balance: {:.2f}, Real P/L: {:.2f}, Mean Real P/L: {:.2F}".format(
                    self.end_info["reward"], self.end_info["orders"], self.end_info["mean_reward"], self.end_info["profit_factor"], self.end_info["real_profit_factor"], self.end_info["mean_target"], self.end_info["mean_risk"], self.end_info["mean_bars"], self.end_info["balance"], self.end_info["max_balance"], self.end_info["total_pl"], self.end_info["mean_pl"], self.end_info["real_orders"], self.end_info["real_balance"], self.end_info["max_real_balance"], self.end_info["real_pl"], self.end_info["mean_real_pl"]))

            return True

        return False

    def gen_info(self):
        if self.total_orders <= 0:
            self.total_orders = 1

        if self.real_orders <= 0:
            self.real_orders = 1

        self.end_info = {
            "reward": self.total_reward,
            "orders": self.total_orders,
            "mean_reward": self.total_reward/self.total_orders,
            "profit_factor": self.balance / self.initial_balance,
            "real_profit_factor": self.real_balance/self.initial_balance,
            "mean_target": self.total_target / self.total_orders,
            "mean_risk": self.total_risk / self.total_orders,
            "mean_bars": self.total_bars / self.total_orders,
            "balance": self.balance,
            "max_balance": self.max_balance,
            "total_pl": self.total_pl,
            "mean_pl": self.total_pl / self.total_orders,
            "real_orders": self.real_orders,
            "real_balance": self.real_balance,
            "max_real_balance": self.max_real_balance,
            "real_pl": self.real_pl,
            "mean_real_pl": self.real_pl/self.real_orders
        }

    def log(self, prefix, msg, debug_level=2):
        if debug_level <= self.debug_mode:
            print("[{}] {}".format(prefix, msg))

    def render(self, mode='human'):
        pass

    def convert_action(self, action):
        target_percentage = action[0]
        risk_percentage = action[1]

        if target_percentage != 0:
            if target_percentage > 0:
                base_points = self.min_target_points
            elif target_percentage < 0:
                base_points = -self.min_target_points

            risk = self.min_risk_ratio + \
                ((self.max_risk_ratio - self.min_risk_ratio)*risk_percentage)

            target_points = base_points + \
                (target_percentage*(self.max_target_points-self.min_target_points))

            tp = target_points*.00001
            sl = target_points*risk*.00001

            return tp, sl
        else:
            return 0, 0
