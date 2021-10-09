"""
V5: Hellbent

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


class Hellbent(gym.Env):
    def __init__(self, history_path, max_target_points=100, min_risk=0.5, max_risk=3, drawdown_limit=250, debug_mode=1):
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
        self.max_target_points = max_target_points
        self.min_risk = min_risk
        self.max_risk = max_risk
        self.drawdown_limit = -(drawdown_limit*.00001)
        self.debug_mode = debug_mode

        # Load CSV files
        # self.full_history = []
        file = open(os.path.join(history_path, "meta.txt"), "r")
        self.history_chunks = int(file.read())

        self.log("Init", "Found {} history files".format(
            self.history_chunks), 1)

        # for i in range(self.history_chunks):
        #     self.full_history.append(pd.read_csv(
        #         os.path.join(history_path, "{:d}.csv".format(i))))

        # self.log("Init", "History loaded succesfully", 1)

    def reset(self):
        self.total_reward = 0.0
        self.total_trade_reward = 0.0
        self.total_bars_held = 0
        self.total_risk = 0
        self.drawdown = 0.0
        self.total_trades = 0
        self.done = False

        index = random.randint(0, self.history_chunks)
        #self.history = self.full_history[index].to_numpy()
        self.history = pd.read_csv(os.path.join(
            self.history_path, "{}.csv".format(index))).to_numpy()
        self.history_size = self.history.shape[0]
        self.current_pos = 0

        self.log("Reset", "Loaded history file {}".format(index), 1)

        return self.get_observation()

    def step(self, action):
        target = round(action[0]*self.max_target_points)
        risk = self.min_risk + (action[1]*(self.max_risk-self.min_risk))

        self.total_risk += risk

        if target == 0:
            obs = self.get_observation()
            self.check_done()
            return obs, 0, self.done, {}

        # Calculate reference price (open price)
        if target > 0:
            open_price = self.bid
        elif target < 0:
            open_price = self.ask

        tp = round(open_price + (target*.00001), 5)
        sl = round(open_price - ((target*.00001)*risk), 5)

        long_trade = False
        short_trade = False

        if tp > sl:  # close long trade
            long_trade = True
            close_ref = BID_INDEX
        elif tp < sl:  # close short trade
            short_trade = True
            close_ref = ASK_INDEX

        close_price, bars_held = self.find_close_price(tp, sl, close_ref)

        trade_reward = 0

        if long_trade:
            trade_reward = (close_price - open_price)
        elif short_trade:
            trade_reward = (open_price - close_price)

        #hold_penalty = bars_held*.00002
        hold_modifier = ((15 - bars_held)/20)*trade_reward

        reward = 0

        if trade_reward > 0:
            reward = trade_reward + hold_modifier
            risk_bonus = 0.1 * (1 - action[0])
            reward += (risk_bonus*trade_reward)
        elif trade_reward < 0:
            reward = trade_reward + (trade_reward*(bars_held/20))
            risk_bonus = 0.1 * action[0]
            reward += (risk_bonus*trade_reward)

        #print(action, trade_reward, hold_penalty, risk, reward)

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
        self.check_done()
        return obs, reward, self.done, {}

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

        return obs

    def check_done(self):
        if self.done:
            self.log("Episode End", "Reward: {:.5f}, Trade Reward: {:.0f}, Trades: {:d}, Mean Trw/Tr: {:.2f}, Mean Bars/Trade: {:.2f}, Mean Risk: {:.2f}, Final Drawdown: {:.0f}".format(
                self.total_reward, self.total_trade_reward*100000, self.total_trades, self.total_trade_reward*100000/self.total_trades, self.total_bars_held/self.total_trades, self.total_risk/self.total_trades, self.drawdown*100000), 1)

    def log(self, prefix, msg, debug_level=2):
        if debug_level <= self.debug_mode:
            print("[{}] {}".format(prefix, msg))

    def render(self, mode='human'):
        pass
