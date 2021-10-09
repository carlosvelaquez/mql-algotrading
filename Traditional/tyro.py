import gym
import pandas as pd
import numpy as np

from gym import spaces

class TyroEnv(gym.Env) :
    window_size = 100
    data = None
    index = 0
    considered = None

    balance = 0
    tp = 50
    sl = 40

    ask = 0.0
    bid = 0.0

    orders = []

    def __init__(self):
        super(TyroEnv, self).__init__()

        df = pd.read_csv("dataset.csv")
        self.data = df[["Month","Day","Hour","Minute","Open","High","Low","Close","Spread","Volume","ema5","ema10","ema25","ema50","ema100","ema200","sma5","sma10","sma25","sma50","sma100","sma200","bands10","bands20","bands40","bands80","bands160","sar1","sar2","sar4","sar8","sar16","sar32","macd6a","macd6b","macd12a","macd12b","macd24a","macd24b","macd48a","macd48b","macd92a","macd92b","macd184a","macd184b","rsi7","rsi14","rsi30","rsi60","rsi120","rsi250","stoch5a","stoch5b","stoch14a","stoch14b","stoch21a","stoch21b","stoch42a","stoch42b","stoch84a","stoch84b","adx7","adx14","adx28","adx52","adx104","adx208","momentum7","momentum14","momentum28","momentum52","momentum104","momentum208","mfi7","mfi14","mfi28","mfi56","mfi112","mfi224","atr5","atr10","atr20","atr40","atr80","atr160"]]

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        self.action_space = spaces.Discrete(3) # Long, Short, Hold

        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    
    def update_orders(self):
        for order in self.orders:
            if order.type == 1: #long
                if self.bid <= order.sl:
                    balance -= (order.price - )
                    self.orders.remove(order)

                elif self.ask >= order.tp:
                    self.orders.remove(order)

            elif order.type == 2: #short
                pass

    def next_observation(self):
        index = self.index
        window_size = self.window_size

        price = self.data[index]["Open"]
        spread = self.data[index]["Spread"]

        ask = price + ((spread/2)*.00001)
        bid = price - ((spread/2)*.00001)

        return self.data[index : index + window_size]

    def act(self, action):
        ask = self.ask
        bid = self.bid

        if action == 1:
            order = {
                "price" : ask,
                "tp" : ask + (self.tp*.00001),
                "sl" : bid - (self.sl*.00001),
                "type" : action
            }
        elif action == 2:
            order = {
                "price": bid,
                "tp": bid - (self.tp*.00001),
                "sl": ask + (self.sl*.00001),
                "type" : action
            }


    def step(self, action):
        self.index += 1
        return next_observation()

    def reset(self):
        self.balance = 0
        self.index = np.random.randint(0, self.data.shape[0] - self.window_size - 1)

        return next_observation()


    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

