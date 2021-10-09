"""
V6: Stellar

Observations: Open, High, Low, Close, Tick Volume, MA5, MA15, MA50, MA100, EMA5, EMA15, EMA50, EMA100, RSI, Stoch1, Stoch2, BBU, BBM, BBD, Bid, Ask
Actions: Continuous, (TP, SL)

"""

import os
import gym
import random
import datetime
import numpy as np
import pandas as pd

from gym import spaces

NUM_FEATURES = 21
BID_INDEX = 19
ASK_INDEX = 20


class Stellar(gym.Env):
    def __init__(self, history_path, min_target_points=25, max_target_points=100, min_risk=1, max_risk=3, max_bars_per_trade=15, drawdown_limit=500, normalize=True, disable_done=False, debug_mode=1):
        random.seed(datetime.datetime.now())

        # Spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([2.0, 2.0, 2.0, 2.0, 250, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 100, 100, 100, 2.0, 2.0, 2.0, 2.0, 2.0]))

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.01]), high=np.array([1.0, 1.0]))

        # Parameters
        self.history_path = history_path
        self.min_target_points = min_target_points
        self.max_target_points = max_target_points
        self.min_risk = min_risk
        self.max_risk = max_risk
        self.max_bars_per_trade = max_bars_per_trade
        self.drawdown_limit = -(drawdown_limit*.00001)
        self.normalize = normalize
        self.disable_done = disable_done
        self.debug_mode = debug_mode

        # Load CSV files
        file = open(os.path.join(history_path, "meta.txt"), "r")
        self.history_chunks = int(file.read())

        self.log("Init", "Found {} history files".format(
            self.history_chunks), 1)

    def reset(self):
        self.total_reward = 0.0
        self.total_trade_reward = 0.0
        self.total_bars_held = 0
        self.total_risk = 0
        self.drawdown = 0.0
        self.total_trades = 0
        self.done = False

        index = random.randint(0, self.history_chunks)

        self.history = pd.read_csv(os.path.join(
            self.history_path, "{}.csv".format(index))).to_numpy()
        self.history_size = self.history.shape[0]
        self.current_pos = 0

        #self.log("Reset", "Loaded history file {}".format(index), 1)

        self.history_file = index

        return self.get_observation()

    def step(self, action):
        target_percentage = action[0]
        risk_percentage = action[1]

        target_points = round(
            target_percentage*(self.max_target_points-self.min_target_points))
        risk = self.min_risk + (risk_percentage*(self.max_risk-self.min_risk))

        self.total_risk += risk

        if target_points == 0:
            obs = self.get_observation()
            self.check_done()
            return obs, 0, self.done, {}

        # Calculate open prices
        if target_points > 0:
            tp_open = self.bid
            sl_open = self.ask
        elif target_points < 0:
            tp_open = self.ask
            sl_open = self.bid

        tp = round(tp_open + (target_points*.00001), 5)
        sl = round(sl_open - ((target_points*.00001)*risk), 5)

        long_trade = False
        short_trade = False

        if tp > sl:  # close long trade
            long_trade = True
            close_ref = BID_INDEX
        elif tp < sl:  # close short trade
            short_trade = True
            close_ref = ASK_INDEX
        else:
            return self.get_observation(), 0, self.done, {}

        close_price, bars_held = self.find_close_price(tp, sl, close_ref)

        trade_reward = 0

        if long_trade:
            trade_reward = (close_price - tp_open)
        elif short_trade:
            trade_reward = (tp_open - close_price)

        if trade_reward > 0:
            hold_modifier = (self.max_bars_per_trade -
                             bars_held)/self.max_bars_per_trade

            if hold_modifier < 0:
                hold_modifier = 0

            risk_reward = trade_reward*(1-risk_percentage)
        else:
            hold_modifier = bars_held/self.max_bars_per_trade
            risk_reward = trade_reward*risk_percentage

        reward = (trade_reward + risk_reward)*(hold_modifier + 1)

        self.total_reward += reward
        self.total_trade_reward += trade_reward
        self.drawdown += trade_reward
        self.total_bars_held += bars_held
        self.total_trades += 1

        if self.drawdown > 0:
            self.drawdown = 0

        if self.drawdown <= self.drawdown_limit:
            self.done = True

        obs = self.get_observation()

        if self.disable_done:
            self.done = False
        else:
            self.check_done()

        return obs, reward, self.done, {"tr": trade_reward}

    def find_close_price(self, x, y, ref_index):
        bars_held = 0

        if x > y:
            upper_bound = x
            lower_bound = y
        elif y > x:
            upper_bound = y
            lower_bound = x

        prev_open = self.history[self.current_pos][0]

        for i in range(self.current_pos, self.history_size - 2):
            ref_price = self.history[i][ref_index]

            if prev_open != self.history[i][0]:
                bars_held += 1
                prev_open = self.history[i][0]

            if ref_price >= upper_bound:
                return upper_bound, bars_held

            if ref_price <= lower_bound:
                return lower_bound, bars_held

        return self.history[self.history_size - 2][ref_index], bars_held

    def get_observation(self):
        obs = self.history[self.current_pos]
        self.current_pos += 1

        self.bid = obs[BID_INDEX]
        self.ask = obs[ASK_INDEX]

        if self.current_pos >= self.history_size:
            self.done = True

        if self.normalize:
            nobs = []

            for i in range(obs.shape[0]):
                nobs.append(obs[i]/self.observation_space.high[i])
            obs = np.array(nobs)

        return obs

    def check_done(self):
        if self.done:
            self.log("Episode End", "R: {:.5f}, TR: {:.0f}, Trades: {:d}, Mean TR: {:.2f}, Mean Bars/Trade: {:.2f}, Mean Risk: {:.2f}, Final Drawdown: {:.0f}, HF: {:d}".format(
                self.total_reward, self.total_trade_reward*100000, self.total_trades, self.total_trade_reward*100000/self.total_trades, self.total_bars_held/self.total_trades, self.total_risk/self.total_trades, self.drawdown*100000, self.history_file), 1)

    def log(self, prefix, msg, debug_level=2):
        if debug_level <= self.debug_mode:
            print("[{}] {}".format(prefix, msg))

    def render(self, mode='human'):
        pass
