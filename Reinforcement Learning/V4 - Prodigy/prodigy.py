"""
V4: Prodigy

Observations: OHLC; Volume; SMA 5,15,50,100; BB; Bid, Ask; Short Price, Long Price over a time period
Actions: Short, Close Short, Hold, Long, Close Long
Reward: P/L when closing an order

Execution Flow:
Init
Config and launch terminal
Wait for Terminal
Terminal queues observation
Step takes observation and sends action
Terminal receives and executes answer
Finish when no messages from terminal or stopped out

"""

import gym
import zmq
import os
import random
import datetime
import numpy as np
import pandas as pd

from gym import spaces

# ACTION_SHORT = 0
# ACTION_CLOSE_SHORT = 1
# ACTION_HOLD = 2
# ACTION_LONG = 3
# ACTION_CLOSE_LONG = 4
ACTION_SHORT = 0
ACTION_HOLD = 1
ACTION_LONG = 2

NUM_FEATURES = 16

BID_INDEX = 12
ASK_INDEX = 13
SHORT_INDEX = 14
LONG_INDEX = 15


class Prodigy(gym.Env):
    def __init__(self, mt5_path, csv_path=None, single_step=False, history_timesteps=360, advance_days=1, initial_balance=200, leverage=2000, min_date=datetime.date(2010, 1, 1), max_date=datetime.date(2020, 1, 1), seconds_per_update=5, drawdown_limit=250, scale_observations=False, debug_mode=1):
        # Protocol stuff
        random.seed(datetime.datetime.now())

        if mt5_path == None and csv_path != None:
            self.full_history = pd.read_csv(csv_path).to_numpy()
            if debug_mode >= 1:
                print("Loaded csv file with shape", self.full_history.shape)
            self.csv_mode = True
        else:
            self.mt5_path = mt5_path
            self.working_dir = os.getcwd()

            # ZMQ Socket
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PAIR)
            self.socket.bind("tcp://*:5555")

            self.csv_mode = False

        # Spaces
        #self.action_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(history_timesteps, NUM_FEATURES))

        # Parameters
        self.single_step = single_step
        self.history_timesteps = history_timesteps
        self.advance_days = advance_days
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.min_date = min_date
        self.max_date = max_date
        self.seconds_per_update = seconds_per_update
        self.drawdown_limit = -drawdown_limit
        self.scale_observations = scale_observations
        self.debug_mode = debug_mode

        if self.single_step:
            self.history_timesteps = 2

        self.max_hold_steps = (10*60)/self.seconds_per_update

    def reset(self):
        self.done = False
        self.history = np.zeros(shape=(self.history_timesteps, NUM_FEATURES))
        self.actions = np.zeros(shape=(self.action_space.n,))
        self.total_reward = 0
        self.total_trade_reward = 0
        self.total_exploration_reward = 0
        self.drawdown = 0

        if self.csv_mode:
            self.reset_csv_mode()
            self.short_order = 0.0
            self.long_order = 0.0
        else:
            self.restart_terminal()
            self.warmup()

        return self.get_observation()

    def reset_csv_mode(self):
        start_index = random.randint(
            0, self.full_history.shape[0] - ((self.advance_days*24*60*60)/self.seconds_per_update) - 1)

        self.current_pos = start_index
        self.final_index = start_index + \
            ((self.advance_days*24*60*60)/self.seconds_per_update) - 1
        history = []

        for i in range(start_index, start_index + self.history_timesteps + 1):
            nrow = self.full_history[i]
            nrow = np.append(nrow, [0.0, 0.0])

            history.append(nrow)
            self.current_pos += 1

        self.history = np.array(history)

        if self.debug_mode >= 2:
            print("History shape: ", self.history.shape)
            print(self.history)

    def restart_terminal(self):
        self.log("Terminal", "Starting MT5 Terminal...", 1)

        os.system("pwsh.exe -c \"taskkill /F /IM terminal64.exe /T\"")
        self.gen_mt5_config()

        start_command = "pwsh.exe -c \"start '{}' /config:'{}\mt5_config.ini'\"".format(
            self.mt5_path, self.working_dir)

        self.log("Terminal", "Start command: {}".format(start_command), 3)
        os.system(start_command)

        self.connect_to_terminal()
        self.send_terminal_params()

    def gen_mt5_config(self):
        file = open('mt5_config.ini', 'w')
        start_date, end_date = self.get_random_date_range()

        file.write("[Tester]\nExpert=ProdigyInterface\nSymbol=EURUSD\nPeriod=M1\nModel=0\nOptimization=0\nFromDate={}\nToDate={}\nShutdownTerminal=0\nDeposit={}\nCurrency=USD\nLeverage=1:{}\nVisual=0".format(
            start_date, end_date, self.initial_balance, self.leverage))
        file.close()

    def get_random_date_range(self):
        time_between_dates = self.max_date - self.min_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)

        start_date = self.min_date + \
            datetime.timedelta(days=random_number_of_days)

        end_date = start_date + datetime.timedelta(days=self.advance_days)

        start_date_str = "{}.{}.{}".format(
            start_date.year, start_date.month, start_date.day)

        end_date_str = "{}.{}.{}".format(
            end_date.year, end_date.month, end_date.day)

        return start_date_str, end_date_str

    def connect_to_terminal(self, retries=3, timeout=None):
        msg = self.socket.recv().decode("utf-8")

        if msg == "ready":
            self.log("Reset", "MT5 interface ready", 1)
        else:
            self.log(
                "Reset", "Error with MT5 interface (received {}), retrying...".format(msg), 0)

            msg = self.socket.recv().decode("utf-8")

            if msg == "ready":
                self.log("Reset", "MT5 interface ready", 1)
            else:
                self.log(
                    "Reset", "Error with MT5 interface (received {})".format(msg), 0)

    def send_terminal_params(self):
        events = self.socket.poll(timeout=5000, flags=zmq.POLLOUT)

        if events > 0:
            self.socket.send_string("{}".format(self.seconds_per_update))
        else:
            self.log("FATAL", "Timeout sending parameters to MT5 Terminal", 0)
            exit()

    def warmup(self):
        history = []

        for i in range(self.history_timesteps):
            self.send_action(ACTION_HOLD)
            data = self.receive_row()

            if data == False:
                self.log(
                    "Warm-Up", "Warming up failed. Terminal stopped sending data.", 1)
                self.reset()
                return

            vals = self.cast(data.split(","))
            history.append(vals)

        self.history = np.array(history)

        if self.debug_mode >= 2:
            print("History shape:", self.history.shape)
            print(self.history)

    def cast(self, arr):
        casted = []

        for x in arr:
            casted.append(float(x))

        return casted

    def receive_row(self):
        events = self.socket.poll(timeout=5000, flags=zmq.POLLIN)

        self.log("Receive Poll", "Events: {}".format(events), 3)

        if events > 0:
            return self.socket.recv().decode("utf-8")
        else:
            return False

    def step(self, action):
        if self.csv_mode:
            self.csv_act(action)
        else:
            if not self.send_action(action):
                self.done = True

        if not self.done:
            self.advance()

        obs = self.get_observation()
        reward = self.calc_reward()

        if self.drawdown < self.drawdown_limit:
            self.done = True

        self.check_done()

        return obs, reward, self.done, {}

    def csv_act(self, action):
        self.actions[action] += 1

        if action == ACTION_SHORT:
            if self.short_order == 0.0:
                self.short_order = self.history[-1, BID_INDEX]
            else:
                self.short_order = 0

        if action == ACTION_LONG:
            if self.long_order == 0.0:
                self.long_order = self.history[-1, ASK_INDEX]
            else:
                self.long_order = 0

    def send_action(self, action):
        self.actions[action] += 1

        events = self.socket.poll(timeout=5000, flags=zmq.POLLOUT)

        self.log("Send Poll", "Events: {}".format(events), 3)
        self.log("Step", "Sending action: {}".format(action), 3)

        if events > 0:
            self.socket.send_string(str(action))
            return True
        else:
            return False

    def advance(self):
        nvals = []

        if self.csv_mode:
            nvals = self.get_csv_row()
        else:
            data = self.receive_row()

            if not data:
                self.log(
                    "Advance", "Terminal stopped sending data; ending episode.", 1)
                self.done = True
                return

            nvals = self.cast(data.split(","))

        arr = np.delete(self.history, obj=0, axis=0)
        nrow = np.array(nvals, ndmin=2)

        self.history = np.append(arr, nrow, axis=0)

    def get_csv_row(self):
        row = self.full_history[self.current_pos]
        row = np.append(row, self.short_order)
        row = np.append(row, self.long_order)

        self.current_pos += 1

        if self.current_pos >= self.final_index:
            self.done = True

        return row

    def check_done(self):
        if self.done:
            self.log("Episode End", "Total reward: {:.2f} | Trade reward: {:.2f} | Exp reward: {:.2f} | Final drawdown: {:.2f} ".format(
                self.total_reward, self.total_trade_reward, self.total_exploration_reward, self.drawdown), 1)

            if self.debug_mode >= 1:
                print("Actions on the previous episode:")
                print(self.actions)

    def get_observation(self):
        obs = np.copy(self.history)

        if self.scale_observations:
            obs /= 2

        if self.single_step:
            obs = np.reshape(obs[-1], (NUM_FEATURES,))

        return obs

    def calc_reward(self):
        short_pl = 0
        long_pl = 0
        exploration_reward = 0

        prev_short = self.history[-2, SHORT_INDEX]
        new_short = self.history[-1, SHORT_INDEX]
        prev_long = self.history[-2, LONG_INDEX]
        new_long = self.history[-1, LONG_INDEX]

        bid = self.history[-2, BID_INDEX]
        ask = self.history[-2, ASK_INDEX]

        # Conditions
        opened_short = (prev_short == 0.0) and (new_short != 0.0)
        opened_long = (prev_long == 0.0) and (new_long != 0.0)
        closed_short = (prev_short != 0.0) and (new_short == 0.0)
        closed_long = (prev_long != 0.0) and (new_long == 0.0)
        held_short = (prev_short != 0.0) and (prev_short == new_short)
        held_long = (prev_long != 0.0) and (prev_long == new_long)

        if closed_short:
            short_pl = prev_short - ask
            exploration_reward -= 4

        if closed_long:
            long_pl = bid - prev_long
            exploration_reward -= 4

        if opened_short:
            exploration_reward += 5
            self.short_hold = 0

        if opened_long:
            exploration_reward += 5
            self.long_hold = 0

        if held_short:
            if self.short_hold <= self.max_hold_steps:
                exploration_reward += .01
                self.short_hold += 1

        if held_long:
            if self.long_hold <= self.max_hold_steps:
                exploration_reward += .01
                self.long_hold += 1

        trade_reward = (short_pl + long_pl)*100000
        self.drawdown += trade_reward

        if self.drawdown > 0:
            self.drawdown = 0

        reward = exploration_reward + trade_reward

        self.total_reward += reward
        self.total_exploration_reward += exploration_reward
        self.total_trade_reward += trade_reward

        return reward

    def log(self, prefix, msg, debug_level=2):
        if debug_level <= self.debug_mode:
            print("[{}] {}".format(prefix, msg))

    def render(self, mode='human'):
        pass
