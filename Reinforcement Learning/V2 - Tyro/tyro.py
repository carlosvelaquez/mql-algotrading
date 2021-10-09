import random
import gym

import numpy as np
import pandas as pd

from gym import spaces
from datetime import datetime

SHORT = 0
CLOSE_SHORT = 1
HOLD = 2
LONG = 3
CLOSE_LONG = 4


class Tyro(gym.Env):
    def __init__(self, history_file_path, base_currency="EUR", quote_currency="USD", account_currency="USD", initial_balance=1000, leverage=100, stop_out_level=0.5, history_timesteps=600, ticks_per_episode=1800, fixed_position_size=0.01, risk=None, pip_risk=5, lot_size=100000, min_position_size=0.01, debug_mode=0):
        # Constant stuff
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.account_currency = account_currency
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.stop_out_level = stop_out_level
        self.history_timesteps = history_timesteps
        self.ticks_per_episode = ticks_per_episode
        self.fixed_position_size = fixed_position_size
        self.risk = risk
        self.pip_risk = pip_risk
        self.lot_size = lot_size
        self.min_position_size = min_position_size

        self.DEBUG = debug_mode

        # Observation: OHLC Values, spread, bid, ask, volume, balance, equity, short_order, long_order, used_margin, free_margin
        self.observation_space = spaces.Box(
            # low=99999, high=99999, shape=(history_timesteps, 14))
            low=-99999, high=99999, shape=(history_timesteps*14,))

        # Actions: SHORT, CLOSE_SHORT, HOLD, LONG, CLOSE_LONG
        self.action_space = spaces.Discrete(5)

        # Load history
        self.full_history = self.load_history_file(history_file_path)
        self.full_history_length = self.full_history.shape[0]

        if self.full_history_length < self.history_timesteps + self.ticks_per_episode:
            print("Full history length: %i, Minimum history length: %i" % (
                self.full_history_length, self.history_timesteps + self.ticks_per_episode))
            raise Exception(
                "Full history is too short for specified parameters.")

        random.seed(datetime.now())
        self.reset()
        self.print_init_info()

    def print_init_info(self):
        print("-----\nInitialized Tyro environment with the following parameters:\n")
        print("Pair: {}/{}, account currency: {}".format(self.base_currency,
                                                         self.quote_currency, self.account_currency))
        print("Initial Balance: {} {}".format(
            self.initial_balance, self.account_currency))
        print("Leverage: 1:{}".format(self.leverage))
        print("Stop out level: {}%".format(int(self.stop_out_level*100)))
        print("History timesteps: {}".format(self.history_timesteps))
        print("Ticks per episode: {}".format(self.ticks_per_episode))
        print("Fixed position size: {}".format(self.fixed_position_size))
        if self.risk:
            print("Risk percentage: {}%".format(self.risk*100))
            print("Pip risk: {}".format(self.pip_risk))
        print("Lot size: {}".format(self.lot_size))
        print("Minimum position size: {}".format(self.min_position_size))
        print("Debug Mode: {}".format(self.DEBUG))
        print("-----\n")

    def load_history_file(self, path):
        fields = ['Open', 'High', 'Low', 'Close',
                  'Spread', 'Bid', 'Ask', 'Volume']
        dataframe = pd.read_csv(path, usecols=fields)

        if self.DEBUG == 2:
            print("Loaded history:")
            print(dataframe)

        return dataframe

    def reset(self):
        self.init_history_arrays()

        self.balance = self.initial_balance
        self.equity = self.initial_balance

        self.short_order = None
        self.long_order = None

        self.calc_margin()

        self.current_tick = self.history_timesteps
        self.tick_count = 0

        return self.get_observation()

    def init_history_arrays(self):
        # Start index is a random number between history_timesteps and the full history length minus the ticks per episode
        self.start_index = random.randint(self.history_timesteps,
                                          self.full_history_length - self.ticks_per_episode - 1)

        # End index is the start index plus the timesteps and ticks per episode
        self.end_index = self.start_index + \
            self.history_timesteps + self.ticks_per_episode

        # Current history window delimited by the calculated indexes
        self.history = self.full_history[self.start_index: self.end_index]

        # Balance and Equity history, size timesteps and initialized with the initial_balance
        self.balance_history = np.full(
            (self.history_timesteps, 1), self.initial_balance)
        self.equity_history = np.full(
            (self.history_timesteps, 1), self.initial_balance)

        # Short and long order prices history, initialized with 0
        self.short_history = np.zeros((self.history_timesteps, 1))
        self.long_history = np.zeros((self.history_timesteps, 1))

        # Used and Free Margin history, used initialized with zero and free with initial_balance
        self.used_margin_history = np.zeros((self.history_timesteps, 1))
        self.free_margin_history = np.full(
            (self.history_timesteps, 1), self.initial_balance)

    def get_observation(self):
        # Observation: OHLC Values, spread, bid, ask, volume, balance, equity, short_order, long_order, used_margin, free_margin
        history_obs = self.history[self.current_tick -
                                   self.history_timesteps: self.current_tick]

        if self.DEBUG == 2:
            print("History Observation Shape:", history_obs.shape)
            print("Balance History Shape:", self.balance_history.shape)
            print("Equity History Shape:", self.equity_history.shape)
            print("Short History Shape:", self.short_history.shape)
            print("Long History Shape:", self.long_history.shape)
            print("Used Margin History Shape:",
                  self.used_margin_history.shape)
            print("Free Margin History Shape:",
                  self.free_margin_history.shape)

        obs = np.concatenate((history_obs, self.balance_history,
                              self.equity_history, self.short_history, self.long_history, self.used_margin_history, self.free_margin_history), axis=1)

        if self.DEBUG == 2:
            print("Observation Shape:", obs.shape)
            print(obs)

        # return obs
        return obs.flatten()

    def step(self, action):
        prev_balance = self.balance

        self.current_tick += 1
        self.tick_count += 1

        current_vals = self.history.iloc[self.current_tick]
        self.bid = current_vals['Bid']
        self.ask = current_vals['Ask']

        self.act(action)
        self.equity = self.balance + self.calc_floating_pl()
        self.calc_margin()

        if self.DEBUG == 1:
            print("Balance: %5f, Equity: %5f" % (self.balance, self.equity))

        self.advance_history()

        done = False
        if self.tick_count >= self.ticks_per_episode - 1:
            if self.DEBUG == 1:
                print("Episode max ticks reached, ending...")
            done = True

        if self.margin_level <= self.stop_out_level:
            if self.DEBUG == 1:
                print("Stopping out at Margin Level %i percent, Stop out level: %i percent" % (
                    int(self.margin_level*100), int(self.stop_out_level*100)))
            done = True

        obs = self.get_observation()
        reward = prev_balance - self.balance

        if self.DEBUG == 2:
            self.render()

        return obs, reward, done, {}

    def act(self, action):
        # Calculate position size if creating an order
        if action == SHORT or action == LONG:
            pos_size = self.calc_position_size()

        if action == SHORT:
            if not self.short_order:
                self.short_order = {
                    "size": pos_size,
                    "price": self.bid
                }

                if self.DEBUG == 2:
                    print("Placing SHORT order... Position size: %i, at a price of %5f" % (
                        pos_size, self.bid))
            else:
                if self.DEBUG == 2:
                    print("Received SHORT action, but a short order exists already.")

        elif action == CLOSE_SHORT:
            if self.short_order:
                size = self.short_order["size"]
                price = self.short_order["price"]

                pl = size * (self.ask - price)
                self.balance += pl
                self.short_order = None

                if self.DEBUG == 2:
                    print("Closing SHORT order... Position size: %i, at a P/L of %5f" % (
                        size, pl))
            else:
                if self.DEBUG == 2:
                    print(
                        "Received CLOSE_SHORT action, but there are no open short orders.")

        elif action == LONG:
            if not self.long_order:
                self.long_order = {
                    "size": pos_size,
                    "price": self.ask
                }

                if self.DEBUG == 2:
                    print("Placing LONG order... Position size: %i, at a price of %5f" % (
                        pos_size, self.bid))
            else:
                if self.DEBUG == 2:
                    print("Received LONG action, but a long order exists already.")

        elif action == CLOSE_LONG:
            if self.long_order:
                size = self.long_order["size"]
                price = self.long_order["price"]

                pl = size * (self.bid - price)
                self.balance += pl
                self.long_order = None

                if self.DEBUG == 2:
                    print("Closing LONG order... Position size: %i, at a P/L of %4f" % (
                        size, pl))
            else:
                if self.DEBUG == 2:
                    print(
                        "Received CLOSE_LONG action, but there are no open long orders.")

    def calc_position_size(self):
        pos_size = int(0.001*100000)
        min_pos = self.min_position_size*self.lot_size

        if self.fixed_position_size:
            pos_size = int(self.fixed_position_size*self.lot_size)
        elif self.risk:
            pos_size = int((self.balance*self.risk)/(self.pip_risk*.0001))

        if pos_size < min_pos:
            pos_size = min_pos

        return pos_size

    def calc_floating_pl(self):
        pl = 0

        if self.short_order:
            size = self.short_order["size"]
            price = self.short_order["price"]

            pl += size * (self.ask - price)

        if self.long_order:
            size = self.long_order["size"]
            price = self.long_order["price"]

            pl += size * (self.bid - price)

        if self.DEBUG == 2:
            print("Floating P/L:", pl)

        return pl

    def calc_margin(self):
        self.used_margin = 0

        if self.short_order:
            notional_value = self.short_order["size"]

            if not (self.base_currency == self.account_currency):
                notional_value *= self.short_order["price"]

            self.used_margin += (notional_value * (1/self.leverage))

        if self.long_order:
            notional_value = self.long_order["size"]

            if not (self.base_currency == self.account_currency):
                notional_value *= self.long_order["price"]

            self.used_margin += (notional_value * (1/self.leverage))

        self.free_margin = self.equity - self.used_margin

        if self.used_margin != 0:
            self.margin_level = self.equity / self.used_margin
        else:
            self.margin_level = 9.99

        if self.DEBUG == 1:
            print("Used Margin: %4f, Free Margin: %4f, Margin Level: %i percent" %
                  (self.used_margin, self.free_margin, int(self.margin_level*100)))

    def advance_history(self):
        np.delete(self.balance_history, obj=0, axis=0)
        np.append(self.balance_history, self.balance)

        np.delete(self.equity_history, obj=0, axis=0)
        np.append(self.equity_history, self.equity)

        # Drop first row and append order prices (if existent)
        np.delete(self.short_history, obj=0, axis=0)
        if self.short_order:
            np.append(self.short_history, self.short_order["price"])
        else:
            np.append(self.short_history, 0)

        np.delete(self.long_history, obj=0, axis=0)
        if self.long_order:
            np.append(self.long_history, self.long_order["price"])
        else:
            np.append(self.long_history, 0)

        np.delete(self.used_margin_history, obj=0, axis=0)
        np.append(self.used_margin_history, self.used_margin)

        np.delete(self.free_margin_history, obj=0, axis=0)
        np.append(self.free_margin_history, self.free_margin)

    def render(self, mode='human'):
        short_order_price = 0
        long_order_price = 0

        if self.short_order:
            short_order_price = self.short_order["price"]

        if self.long_order:
            long_order_price = self.long_order["price"]

        print("Bal: {:.4f}, Eq: {:.4f} | UM: {:.4f}, FM: {:.4f}, M Lvl: {:d}% | Shrt: {:.4f}, Lng: {:.4f}\r".format(
            self.balance, self.equity, self.used_margin, self.free_margin, int(self.margin_level*100), short_order_price, long_order_price)),
