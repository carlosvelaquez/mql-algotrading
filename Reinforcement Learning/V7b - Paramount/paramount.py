"""
V7: Paramount

Observations: Open, High, Low, Close, Tick Volume, MA5, MA15, MA50, MA100, EMA5, EMA15, EMA50, EMA100, RSI, Stoch1, Stoch2, BBU, BBM, BBD, Order Open Price, Order TP, Order SL, Order Pos Size, Balance, Equity
Actions: Continuous, (TP, SL)

"""

import os
import gym
import random
import datetime
import numpy as np
import pandas as pd

from gym import spaces

BID_INDEX = 3
ASK_INDEX = 3


class Paramount(gym.Env):
    def __init__(self, history_path="./history", initial_balance=200, floating_reward_modifier=.25, min_target_points=5, max_target_points=100, min_risk=1, max_risk=3, account_risk=.01, max_drawdown_percentage=.05, drawdown_compensation=.25, lot_size=100000, min_pos_size=1000, comission=3.5, disable_done=False, debug_mode=1):
        random.seed(datetime.datetime.now())

        # Spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([2.0, 2.0, 2.0, 2.0, 250, 2.0, 2.0, 2.0, 2.0, 2.0,
                           2.0, 2.0, 2.0, 100, 100, 100, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, np.inf, np.inf, np.inf]))

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]))

        # Parameters
        self.history_path = history_path
        self.initial_balance = initial_balance
        self.floating_reward_modifier = floating_reward_modifier
        self.min_target_points = min_target_points
        self.max_target_points = max_target_points
        self.min_risk = min_risk
        self.max_risk = max_risk
        self.account_risk = account_risk
        self.max_drawdown_percentage = max_drawdown_percentage
        self.drawdown_compensation = drawdown_compensation
        self.lot_size = lot_size
        self.min_pos_size = min_pos_size
        self.comission = comission
        self.disable_done = disable_done
        self.debug_mode = debug_mode

        # Load CSV files
        file = open(os.path.join(history_path, "meta.txt"), "r")
        self.history_chunks = int(file.read())

        self.log("Init", "Found {} history files".format(
            self.history_chunks), 1)

    def reset(self):
        # Initialize values
        self.balance = self.initial_balance
        self.equity = self.balance
        self.order = None

        self.total_equity_reward = 0.0
        self.total_trade_reward = 0.0
        self.total_target_points = 0
        self.total_reward = 0
        self.total_trades = 0
        self.total_risk = 0
        self.tick_count = 0
        self.drawdown = 0
        self.prev_pl = 0
        self.done = False

        # Choose a random history file
        index = random.randint(0, self.history_chunks)

        df = pd.read_csv(os.path.join(
            self.history_path, "{}.csv".format(index)))
        df = df.drop(columns=["Bid", "Ask"])
        self.history = df.to_numpy()
        self.history_size = self.history.shape[0]
        self.current_pos = 0

        self.history_file = index

        return self.get_observation()

    def step(self, action):
        target_percentage = action[0]
        risk_percentage = action[1]

        if self.order or (target_percentage == 0):
            reward = self.check_order_close()
            return self.get_observation(), reward, self.done, self.gen_info()
        else:
            self.open_order(target_percentage, risk_percentage)

        reward = self.check_order_close()
        return self.get_observation(), reward, self.done, self.gen_info()

    def open_order(self, target_percentage, risk_percentage):
        if target_percentage > 0:
            base_points = self.min_target_points
        elif target_percentage < 0:
            base_points = -self.min_target_points

        target_points = base_points + \
            (target_percentage*(self.max_target_points-self.min_target_points))
        risk = self.min_risk + \
            ((self.max_risk - self.min_risk)*risk_percentage)

        # Calculate open prices
        if target_points > 0:  # long orders take profit relative to bid
            tp_open = self.bid
            sl_open = self.ask
            close_ref = BID_INDEX

        elif target_points < 0:  # short orders take profit relative to ask
            tp_open = self.ask
            sl_open = self.bid
            close_ref = ASK_INDEX

        tp = round(tp_open + (target_points*.00001), 5)
        sl = round(sl_open - (target_points*.00001*risk), 5)

        pos_size = max(round((self.balance*self.account_risk) /
                             (target_points*.00001*risk)), self.min_pos_size)

        self.order = {
            "open_price": tp_open,
            "tp": tp,
            "sl": sl,
            "pos_size": pos_size,
            "ref_index": close_ref
        }

        self.balance -= self.comission*(pos_size/self.lot_size)
        self.equity = self.balance

        self.total_trades += 1
        self.total_risk += risk
        self.total_target_points += abs(target_points)

    def check_order_close(self):
        reward = 0
        drawdown_stop = False
        history_end_stop = self.current_pos >= self.history_size - 1

        if self.order:
            self.tick_count += 1

            long_order = False
            short_order = False

            open_price = self.order["open_price"]
            tp = self.order["tp"]
            sl = self.order["sl"]
            pos_size = self.order["pos_size"]

            ref_price = self.history[self.current_pos][self.order["ref_index"]]

            if tp > sl:  # Long order
                upper_bound = tp
                lower_bound = sl
                long_order = True
                floating_pl = ref_price - open_price
            elif sl > tp:  # Short Order
                upper_bound = sl
                lower_bound = tp
                short_order = True
                floating_pl = open_price - ref_price

            if floating_pl > 0:
                self.drawdown += floating_pl*self.drawdown_compensation
            else:
                self.drawdown += floating_pl

            drawdown_stop = self.drawdown <= -self.max_drawdown_percentage*self.balance
            bound_reached_stop = (ref_price >= upper_bound) or (
                ref_price <= lower_bound)

            if bound_reached_stop or history_end_stop or drawdown_stop:
                reward = (floating_pl*pos_size) - \
                    ((self.comission*(pos_size/self.lot_size))*2)

                self.balance += reward + \
                    (self.comission*(pos_size/self.lot_size))
                self.equity = self.balance
                self.order = None
                self.prev_pl = 0

                self.total_trade_reward += reward
            else:
                self.equity = self.balance + (floating_pl*pos_size)
                reward = (floating_pl - self.prev_pl) * \
                    self.floating_reward_modifier
                self.prev_pl = floating_pl
                self.total_equity_reward += reward

        if drawdown_stop or history_end_stop:
            self.done = True

        self.total_reward += reward
        return reward

    def get_observation(self):
        obs = self.history[self.current_pos]
        self.current_pos += 1

        if self.order:
            open_price = self.order["open_price"]
            tp = self.order["tp"]
            sl = self.order["sl"]
            pos_size = self.order["pos_size"]
        else:
            open_price = 0
            tp = 0
            sl = 0
            pos_size = 0

        obs = np.append(
            obs, [open_price, tp, sl, pos_size, self.balance, self.equity])

        self.bid = obs[BID_INDEX]
        self.ask = obs[ASK_INDEX]

        self.check_done()
        return obs

    def check_done(self):
        if self.done:
            self.log("Episode End", "Final Balance: {:.2f}, Total Reward: {:.5f}, Trade Reward: {:.2f}, Equity Reward: {:.6f}, Trades: {:d}, Mean TR/Trade: {:.2f}, Mean Ticks/Trade {:.2f}, Mean Target: {:.2f}, Mean Risk: {:.2f}, Final Drawdown: {:.5f}, History File: {:d}".format(
                self.balance, self.total_reward, self.total_trade_reward, self.total_equity_reward, self.total_trades, self.total_trade_reward/self.total_trades, self.tick_count/self.total_trades, self.total_target_points/self.total_trades, self.total_risk/self.total_trades, self.drawdown, self.history_file), 1)

    def gen_info(self):
        return {
            "balance": self.balance,
            "total_trade_reward": self.total_trade_reward,
            "total_target_points": self.total_target_points,
            "total_risk": self.total_risk,
            "total_trades": self.total_trades
        }

    def log(self, prefix, msg, debug_level=2):
        if debug_level <= self.debug_mode:
            print("[{}] {}".format(prefix, msg))

    def render(self, mode='human'):
        pass
