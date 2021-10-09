import random
import gym
import os
import datetime
import zmq

from sklearn import preprocessing

import numpy as np
from gym import spaces

# ACTION_SHORT = 0
# ACTION_CLOSE_SHORT = 1
# ACTION_HOLD = 2
# ACTION_LONG = 3
# ACTION_CLOSE_LONG = 4
ACTION_SHORT = 0
ACTION_HOLD = 1
ACTION_LONG = 2


class Savant(gym.Env):
    def __init__(self, obs_window_timesteps=400, test_days=1, initial_balance=20000, seconds_per_update=10, debug_mode=0):
        # Spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            # low=-10000000, high=10000000, shape=(obs_window_timesteps, 14))
            # low=-10000000, high=10000000, shape=(14,))
            low=-10000000, high=10000000, shape=(int(obs_window_timesteps*14), ))

        self.obs_window_timesteps = obs_window_timesteps
        self.test_days = test_days
        self.initial_balance = initial_balance
        self.seconds_per_update = seconds_per_update

        self.min_date = datetime.date(2010, 1, 1)
        self.max_date = datetime.date(2020, 1, 1)

        self.history = np.zeros(shape=(obs_window_timesteps, 14))

        self.DEBUG = debug_mode

        random.seed(datetime.datetime.now())

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:5555")
        # self.socket.connect("tcp://localhost:5555")

        # self.reset()

    def log(self, prefix="Log", msg="Log msg", debug_level=2):
        if debug_level <= self.DEBUG:
            print("[{}] {}".format(prefix, msg))

    def reset(self):
        self.shorts = 0
        self.holds = 0
        self.longs = 0
        self.restart_terminal()
        return self.get_observation()

    def restart_terminal(self):
        os.system("pwsh.exe -File kill_mt5.ps1")
        time_between_dates = self.max_date - self.min_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)

        start_date = self.min_date + \
            datetime.timedelta(days=random_number_of_days)

        end_date = start_date + datetime.timedelta(days=self.test_days)

        start_date_str = "{}.{}.{}".format(
            start_date.year, start_date.month, start_date.day)

        end_date_str = "{}.{}.{}".format(
            end_date.year, end_date.month, end_date.day)

        file = open('mt5_config.ini', 'w')
        file.write("[Tester]\nExpert=SavantInterface\nSymbol=EURUSD\nPeriod=M1\nModel=0\nOptimization=0\nFromDate={}\nToDate={}\nShutdownTerminal=0\nDeposit={}\nCurrency=USD\nLeverage=1:1000\nVisual=0".format(
            start_date_str, end_date_str, self.initial_balance))
        file.close()

        self.log("Reset", "Starting MT5 Terminal...", 1)
        os.system("pwsh.exe -File start_mt5.ps1")

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

        self.socket.send_string("{}".format(self.seconds_per_update))
        self.warmup()

    def warmup(self):
        history = []

        for i in range(self.obs_window_timesteps):
            self.act(ACTION_HOLD)
            data = self.receive_data()

            if data == False:
                return

            vals = self.cast(data.split(","))
            history.append(vals)

        self.history = np.array(history)

        if self.DEBUG >= 2:
            print("History shape:", self.history.shape)
            print(self.history)

    def act(self, action):
        if action == ACTION_SHORT:
            self.shorts += 1
        elif action == ACTION_HOLD:
            self.holds += 1
        elif action == ACTION_LONG:
            self.longs += 1

        events = self.socket.poll(timeout=5000, flags=zmq.POLLOUT)

        self.log("Send Poll", "Events: {}".format(events), 3)
        self.log("Act", "Executing action: {}".format(action), 3)

        if events > 0:
            self.socket.send_string(str(action))

    def receive_data(self):
        events = self.socket.poll(timeout=5000, flags=zmq.POLLIN)

        self.log("Receive Poll", "Events: {}".format(events), 3)

        if events > 0:
            return self.socket.recv().decode("utf-8")
        else:
            return False

    def query_terminal(self):
        # Observation: OHLC Values, spread, bid, ask, volume, balance, equity, short_order, long_order, used_margin, free_margin
        data = self.receive_data()
        if data == False:
            return True

        arr = np.delete(self.history, obj=0, axis=0)
        nvals = np.array(self.cast(data.split(",")), ndmin=2)

        self.history = np.append(arr, nvals, axis=0)
        return False

    def cast(self, arr):
        casted = []

        for x in arr:
            casted.append(float(x))

        return casted

    def step(self, action):
        prev_balance = float(self.history[-1, 8])
        prev_equity = float(self.history[-1, 9])

        self.act(action)
        done = self.query_terminal()

        new_balance = float(self.history[-1, 8])
        new_equity = float(self.history[-1, 9])

        self.log("Step", "Prev Balance: {}, Prev Equity: {}, New Balance: {}, New Equity: {}".format(
            prev_balance, prev_equity, new_balance, new_equity), 2)

        # reward = (new_balance + (new_equity*.1)) - \
        #     (prev_balance + (prev_equity*.1))

        # reward = new_balance - prev_balance
        reward = new_equity - prev_equity

        if new_balance < 0:
            #reward = -(abs(new_balance) + abs(prev_balance))
            reward = -(abs(new_equity) + abs(prev_equity))

        if done:
            self.log("Episode End", "Shorts: {}, Holds: {}, Longs: {}".format(
                self.shorts, self.holds, self.longs), 1)

        return self.get_observation(), reward, done, {}

    def get_observation(self):
        # return np.reshape(self.history, (self.obs_window_timesteps, 14, 1))
        # return np.copy(self.history).flatten()
        # obs = preprocessing.minmax_scale(np.copy(self.history))
        obs = np.copy(self.history)
        obs = obs.flatten()

        if self.DEBUG >= 2:
            print("Observation")
            print(obs)

        return obs

    def render(self, mode='human'):
        pass
