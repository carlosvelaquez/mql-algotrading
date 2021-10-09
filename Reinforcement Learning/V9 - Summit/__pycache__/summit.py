import os
import gym
import zmq
import random
import numpy as np
import pandas as pd

from enum import IntEnum


class Action(IntEnum):
    HOLD = 0
    SHORT = 1
    CLOSE_SHORT = 2
    LONG = 3
    CLOSE_LONG = 4


class Summit(gym.Env):
    def __init__(self, history_path="./history", deposit=200, lot_size=100000, min_volume=0.01, max_volume=200, commission=7, ref_target_pts=25, acc_risk=0.01, max_acc_loss=0.5, norm_obs=True, debug=1):
        # Parameters
        self.history_path = history_path
        self.deposit = deposit
        self.lot_size = lot_size
        self.min_pos_size = lot_size*min_volume
        self.max_pos_size = lot_size*max_volume
        self.commission = commission
        self.ref_target = ref_target_pts*.00001
        self.acc_risk = acc_risk
        self.max_acc_loss = max_acc_loss
        self.norm_obs = norm_obs
        self.debug = debug

        # Constants
        self.obs_columns = ["Price", "TickVolume", "MA5", "MA15", "MA50", "MA100", "EMA5", "EMA15", "EMA50", "EMA100", "RSI", "Stoch1", "Stoch2", "BBu",
                            "BBm", "BBl", "Shorting", "Longing", "ShortDiff", "LongDiff", "TotalDiff"]
        self.num_features = len(self.obs_columns)

        # Gym Spaces
        self.action_space = gym.spaces.Discrete(len(Action))
        self.observation_space = gym.spaces.Box(
            low=np.array([0.75, 0, 0.75, 0.75, 0.75, 0.75,
                          0.75, 0.75, 0.75, 0.75, 0, 0, 0, 0.75, 0.75, 0.75, 0, 0, -deposit, -deposit, -deposit]),
            high=np.array([1.75, 250, 1.75, 1.75, 1.75, 1.75,
                           1.75, 1.75, 1.75, 1.75, 100, 100, 100, 1.75, 1.75, 1.75, 1, 1, deposit, deposit, deposit])
        )

        # # Reward Scheme
        # self.reward_scheme = {
        #     "open": ref_target_pts/100,
        #     "close": 0,
        #     "hold": ref_target_pts/20000,
        #     "floating_mult": 1/500,
        # }

        # History file count
        file = open(os.path.join(self.history_path, "meta.txt"), "r")
        self.history_file_count = int(file.read())
        file.close()

    def load_history(self):
        index = random.randint(0, self.history_file_count)

        df = pd.read_csv(os.path.join(
            self.history_path, "{}.csv".format(index)))
        df = df.drop(columns=["Open", "High", "Low", "Bid", "Ask"])

        self.history = df.to_numpy()
        self.history_size = self.history.shape[0]
        self.current_history_file = index

    def get_obs(self):
        obs = np.append(self.history[self.current_pos], [
                        self.shorting, self.longing, self.short_diff, self.long_diff, self.floating_diff])

        self.current_pos += 1

        if self.norm_obs:
            obs = (obs - self.observation_space.low) / \
                (self.observation_space.high - self.observation_space.low)

        return obs

    def reset(self):
        # Trackers
        self.total_reward = 0
        self.open_reward = 0
        self.trade_reward = 0
        self.floating_reward = 0
        self.hold_reward = 0

        self.balance = self.deposit
        self.equity = self.deposit
        self.max_equity = self.deposit
        self.min_equity = self.deposit*(1-self.max_acc_loss)

        self.shorting = 0
        self.longing = 0

        self.short_volume = 0
        self.long_volume = 0
        self.short_price = 0
        self.long_price = 0

        self.short_fpl = 0
        self.long_fpl = 0
        self.floating_pl = 0

        self.short_diff = 0
        self.long_diff = 0
        self.floating_diff = 0

        self.short_delta = 0
        self.long_delta = 0
        self.floating_delta = 0

        self.short_trades = 0
        self.long_trades = 0

        self.current_pos = 0

        self.load_history()
        return self.get_obs()

    def calc_pos_size(self):
        pos_size = ((self.balance*self.acc_risk) /
                    self.ref_target)
        pos_size = max(self.min_pos_size, pos_size)
        pos_size = min(self.max_pos_size, pos_size)

        return pos_size

    def step(self, action):
        self.current_price = self.history[self.current_pos, 0]

        if self.short_volume > 0:
            n_diff = self.short_price - self.current_price
            self.short_delta = n_diff - self.short_diff

            self.short_diff = n_diff
            self.short_fpl = n_diff*self.short_volume

        if self.long_volume > 0:
            n_diff = self.current_price - self.long_price
            self.long_delta = n_diff - self.long_diff

            self.long_diff = n_diff
            self.long_fpl = n_diff*self.long_volume

        self.floating_pl = self.short_fpl + self.long_fpl
        self.floating_diff = self.short_diff + self.long_diff
        self.floating_delta = self.short_delta + self.long_delta

        self.equity = self.balance + self.floating_pl

        if self.equity > self.max_equity:
            self.max_equity = self.equity

        equity_calc = self.equity*(1-self.max_acc_loss)

        if equity_calc > self.min_equity:
            self.min_equity = equity_calc

        reward = 0

        if action != Action.HOLD:
            if action == Action.SHORT and self.shorting == 0:
                # reward += self.reward_scheme["open"]
                # self.open_reward += self.reward_scheme["open"]

                pos_size = self.calc_pos_size()
                commissions = self.commission*(pos_size/self.lot_size)

                self.shorting = 1
                self.short_volume = pos_size
                self.short_price = self.current_price
                self.balance -= commissions

            elif action == Action.LONG and self.longing == 0:
                # reward += self.reward_scheme["open"]
                # self.open_reward += self.reward_scheme["open"]

                pos_size = self.calc_pos_size()
                commissions = self.commission*(pos_size/self.lot_size)

                self.longing = 1
                self.long_volume = pos_size
                self.long_price = self.current_price
                self.balance -= commissions

            elif action == Action.CLOSE_SHORT and self.shorting == 1:
                # reward += self.reward_scheme["close"]

                commissions = self.commission

                reward -= commissions
                self.trade_reward -= commissions
                # reward += self.short_diff - commissions
                # self.trade_reward += self.short_diff - commissions
                # reward += self.short_diff
                # self.trade_reward += self.short_diff

                self.balance += self.short_fpl
                self.short_volume = 0
                self.short_price = 0
                self.short_fpl = 0
                self.short_diff = 0
                self.short_delta = 0
                self.short_trades += 1

                self.shorting = 0

            elif action == Action.CLOSE_SHORT and self.longing == 1:
                # reward += self.reward_scheme["close"]

                commissions = self.commission

                reward -= commissions
                self.trade_reward -= commissions
                # reward += self.long_diff - commissions
                # self.trade_reward += self.long_diff - commissions
                # reward += self.long_diff
                # self.trade_reward += self.long_diff

                self.balance += self.long_fpl
                self.long_volume = 0
                self.long_price = 0
                self.long_fpl = 0
                self.long_diff = 0
                self.long_delta = 0
                self.long_trades += 1

                self.longing = 0

        # if self.short_volume > 0:
        #     reward += self.reward_scheme["hold"]
        #     self.hold_reward += self.reward_scheme["hold"]

        # if self.long_volume > 0:
        #     reward += self.reward_scheme["hold"]
        #     self.hold_reward += self.reward_scheme["hold"]

        reward += self.floating_delta
        self.floating_reward += self.floating_delta
        self.total_reward += reward

        return self.get_obs(), reward, self.check_done(), {}

    def check_done(self):
        if self.equity < self.min_equity or self.current_pos >= self.history_size - 1:
            self.log("Episode End",
                     "Reward: {:.5f}, Trade Rw: {:.5f}, Open Rw: {:.5f}, Floating Rw: {:.5f}, Hold Rw: {:.5f}, Balance: {:.2f}, Equity: {:.2f}, Max Equity: {:.2f}, Shorts: {:d}, Longs: {:d}".format(
                         self.total_reward, self.trade_reward, self.open_reward, self.floating_reward, self.hold_reward, self.balance, self.equity, self.max_equity, self.short_trades, self.long_trades
                     ), 1)
            return True

        return False

    def render(self, mode='human'):
        pass

    def log(self, prefix, msg, debug_level=2):
        if debug_level <= self.debug:
            print("[{}] {}".format(prefix, msg))
