import os
import gym
import math
import random
import numpy as np
import pandas as pd

NUM_FEATURES = 19
ENV_FEATURES = 17
MIN_CURRENCY_PRICE = 0.75
MAX_CURRENCY_PRICE = 1.75
MAX_POINT_REWARD = 50000


class Summit(gym.Env):
    def __init__(self, history_path="./history", tensorboard_logdir="./tb", initial_balance=200, pos_size=1000, lot_size=100000, commission=3.5, min_target_points=25, max_target_points=250, min_risk_ratio=.5, max_risk_ratio=3, max_account_loss=.1, max_opt_bars=60, fixed_index=None, normalize_obs=True, debug_mode=1):
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
        self.max_account_loss = max_account_loss
        self.max_opt_bars = max_opt_bars
        self.fixed_index = fixed_index
        self.normalize_obs = normalize_obs
        self.debug_mode = debug_mode

        # Spaces
        low = [MIN_CURRENCY_PRICE for _ in range(NUM_FEATURES)]
        high = [MAX_CURRENCY_PRICE for _ in range(NUM_FEATURES)]

        low[4] = 0  # TickVolume
        low[13] = 0  # RSI
        low[14] = 0  # Stoch1
        low[15] = 0  # Stoch2

        high[4] = 100
        high[13] = 100
        high[14] = 100
        high[15] = 100

        # TotalShortVolume, MaxShortTarget, MinShortTarget, AvgShortTarget, MaxShortRisk, MinShortRisk, AvgShortRisk, ShortPl
        # TotalLongVolume, MaxLongTarget, MinLongTarget, AvgLongTarget, MaxLongRisk, MinLongRisk, AvgLongRisk, LongPl
        # DayPoints

        max_volume = 5000
        min_target = min_target_points*.00001
        max_target = max_target_points*.00001
        max_risk = (max_target_points * max_risk_ratio)*.00001
        min_risk = (min_target_points * min_risk_ratio)*.00001

        env_low = [0, min_target, min_target, min_target, min_risk, min_risk, min_risk, -max_target*max_volume,
                   0, min_target, min_target, min_target, min_risk, min_risk, min_risk, -
                   max_target*max_volume,
                   -max_target*max_volume]

        env_high = [max_volume, max_target, max_target, max_target, max_risk, max_risk, max_risk, max_target*max_volume,
                    max_volume, max_target, max_target, max_target, max_risk, max_risk, max_risk, max_target*max_volume,
                    max_target*max_volume]

        self.observation_space = gym.spaces.Box(
            low=np.append(low, env_low),  high=np.append(high, env_high))

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
        history_obs = self.history[self.current_pos]
        self.current_price = history_obs[3]
        self.current_pos += 1
        self.equity = self.balance + self.floating_pl

        equity_calc = (1 - self.max_account_loss)*self.equity

        if self.last_open != history_obs[0]:
            self.current_bar += 1
            self.last_open = history_obs[0]

        if self.min_equity < equity_calc:
            self.min_equity = equity_calc

        if self.equity > self.max_equity:
            self.max_equity = self.equity

        env_obs = [self.short_volume, self.max_short_target, self.min_short_target, self.total_short_target/self.short_volume,
                   self.max_short_risk, self.min_short_risk, self.total_short_risk /
                   self.short_volume, self.floating_short_pl,
                   self.long_volume, self.max_long_target, self.min_long_target, self.total_long_target/self.long_volume,
                   self.max_long_risk, self.min_long_risk, self.total_long_risk /
                   self.long_volume, self.floating_long_pl,
                   self.day_pl]
        obs = np.append(history_obs, env_obs)

        if self.normalize_obs:
            obs = (obs - self.observation_space.low) / \
                (self.observation_space.high - self.observation_space.low)

        return obs

    def reset(self):
        # Init trackers
        self.current_pos = 0
        self.done = False
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.max_equity = self.balance
        self.min_equity = 0
        self.total_reward = 0
        self.total_pl = 0
        self.total_bars = 0
        self.total_risk = 0
        self.total_target = 0
        self.total_orders = 0
        self.end_info = {}
        self.positions = []
        self.floating_pl = 0
        self.current_bar = 0
        self.last_open = None

        self.short_volume = 0
        self.floating_short_pl = 0
        self.max_short_target = 0
        self.min_short_target = 0
        self.total_short_target = 0
        self.max_short_risk = 0
        self.min_short_risk = 0
        self.total_short_risk = 0

        self.long_volume = 0
        self.floating_long_pl = 0
        self.max_long_target = 0
        self.min_long_target = 0
        self.total_long_target = 0
        self.max_long_risk = 0
        self.min_long_risk = 0
        self.total_long_risk = 0

        self.day_pl = 0

        self.load_random_history_file()
        return self.get_observation()

    def open_position(self, target_percentage, risk_percentage):
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

        # pos_size = max(1000, round((self.equity*self.account_risk_per_trade) /
        #                            (abs(risk_points)*.00001)))
        pos_size = 1000

        self.total_target += abs(target_points)
        self.total_risk += risk

        self.positions.append({
            "tp": tp,
            "sl": sl,
            "open_price": open_price,
            "pos_size": pos_size,
            "open_bar": self.current_bar
        })

    def calc_reward(self):
        reward = 0
        self.floating_pl = 0

        self.short_volume = 0
        self.floating_short_pl = 0
        self.max_short_target = 0
        self.min_short_target = 0
        self.total_short_target = 0
        self.max_short_risk = 0
        self.min_short_risk = 0
        self.total_short_risk = 0

        self.long_volume = 0
        self.floating_long_pl = 0
        self.max_long_target = 0
        self.min_long_target = 0
        self.total_long_target = 0
        self.max_long_risk = 0
        self.min_long_risk = 0
        self.total_long_risk = 0

        for pos in self.positions:
            tp = pos["tp"]
            sl = pos["sl"]
            open_price = pos["open_price"]
            pos_size = pos["pos_size"]
            open_bar = pos["open_bar"]

            target = abs(open_price - tp)
            risk = abs(open_price - sl)

            if tp > sl:
                upper_bound = tp
                lower_bound = sl
                price_diff = self.current_price - open_price

                self.long_volume += 1
                self.floating_long_pl += price_diff
                self.total_long_target += target

                if:
                    pass

            elif sl > tp:
                upper_bound = sl
                lower_bound = tp
                price_diff = open_price - self.current_price
                self.floating_short_pl += price_diff

            commissions = (self.commission*(pos_size/self.lot_size))*2

            floating_pl = (price_diff*pos_size) - commissions

            if self.current_price >= upper_bound or self.current_price <= lower_bound:
                reward += floating_pl
                self.balance += floating_pl
                self.total_pl += floating_pl
                self.total_bars += self.current_bar - open_bar
                self.positions.remove(pos)
            else:
                self.floating_pl += floating_pl

        reward += self.floating_pl*.005
        self.total_reward += reward
        return reward

    def step(self, action):
        target_percentage = action[0]
        risk_percentage = action[1]

        if target_percentage != 0:
            self.open_position(target_percentage, risk_percentage)

        reward = self.calc_reward()
        return self.get_observation(), reward, self.check_done(), self.end_info

    def close_all_positions(self):
        for pos in self.positions:
            tp = pos["tp"]
            sl = pos["sl"]
            open_price = pos["open_price"]
            pos_size = pos["pos_size"]
            open_bar = pos["open_bar"]

            if tp > sl:
                upper_bound = tp
                lower_bound = sl
                price_diff = self.current_price - open_price
            elif sl > tp:
                upper_bound = sl
                lower_bound = tp
                price_diff = open_price - self.current_price

            commissions = (self.commission*(pos_size/self.lot_size))*2

            floating_pl = (price_diff*pos_size) - commissions
            self.balance += floating_pl
            self.total_pl += floating_pl
            self.total_bars += self.current_bar - open_bar
            self.positions.remove(pos)

    def check_done(self):
        done_conditions = []

        done_conditions.append(self.current_pos >= self.history_size - 1)
        done_conditions.append(self.equity <= self.min_equity)

        if True in done_conditions:
            self.close_all_positions()
            self.gen_info()

            if self.debug_mode >= 1:
                print("-----\nReward: {:2f}, Orders: {:d}, Mean Reward: {:.2f}, Profit Factor: {:.2f}\nMean Target: {:.2f}, Mean Risk: {:.2f}, Mean Bars: {:.2f}\nBalance: {:.2f}, Max Equity: {:.2f}, P/L: {:.2f}, Mean P/L: {:.2f}".format(
                    self.end_info["reward"], self.end_info["orders"], self.end_info["mean_reward"], self.end_info["profit_factor"], self.end_info["mean_target"], self.end_info["mean_risk"], self.end_info["mean_bars"], self.end_info["balance"], self.end_info["max_equity"], self.end_info["total_pl"], self.end_info["mean_pl"]))

            return True

        return False

    def gen_info(self):
        if self.total_orders <= 0:
            self.total_orders = 1

        self.end_info = {
            "reward": self.total_reward,
            "orders": self.total_orders,
            "mean_reward": self.total_reward/self.total_orders,
            "profit_factor": self.balance / self.initial_balance,
            "mean_target": self.total_target / self.total_orders,
            "mean_risk": self.total_risk / self.total_orders,
            "mean_bars": self.total_bars / self.total_orders,
            "balance": self.balance,
            "max_equity": self.max_equity,
            "total_pl": self.total_pl,
            "mean_pl": self.total_pl / self.total_orders
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
