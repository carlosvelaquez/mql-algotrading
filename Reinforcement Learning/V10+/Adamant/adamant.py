# V11.0 - Adamant

import os
import gym
import random
import numpy as np
import pandas as pd

from math import log


class Adamant(gym.Env):
    def __init__(self, history_path="./history", obs_timesteps=2000, min_target_pts=25, max_target_pts=250, max_loss=250000, gamma=.99, norm_obs=True, single_position=False, trailing_end=False, debug=1):
        # Parameters
        self.history_path = history_path
        self.obs_timesteps = obs_timesteps
        self.min_target_pts = min_target_pts
        self.max_target_pts = max_target_pts
        self.max_loss = max_loss*.00001
        self.gamma = gamma
        self.norm_obs = norm_obs
        self.single_position = single_position
        self.trailing_end = trailing_end
        self.debug = debug

        self.target_pts_domain = self.max_target_pts-self.min_target_pts
        self.max_bars = 120

        self.obs_low = np.array([0.75, 0.75, 0, 0.75, 0.75, 0.75, 0.75,
                                 0.75, 0.75, 0.75, 0.75, 0, 0, 0, 0.75, 0.75, 0.75])
        self.obs_high = np.array([1.75, 1.75, 250, 1.75, 1.75, 1.75, 1.75,
                                  1.75, 1.75, 1.75, 1.75, 100, 100, 100, 1.75, 1.75, 1.75])

        self.num_features = self.obs_low.shape[0]

        # Gym Spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0])
        )

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_timesteps, self.num_features)
        )

        # Load history info
        file = open(os.path.join(history_path, "meta.txt"), "r")
        self.history_file_count = int(file.read())

        self.log("Initializing", "Found {} history files".format(
            self.history_file_count), 1)

    def load_history(self):
        index = random.randint(0, self.history_file_count)

        df = pd.read_csv(os.path.join(
            self.history_path, "{}.csv".format(index)))
        df = df.drop(columns=["Open", "High", "Low", "Close"])

        self.history = df.to_numpy()
        self.history_size = self.history.shape[0]
        self.current_history_file = index

    def get_obs(self):
        obs = self.history[self.current_pos -
                           self.obs_timesteps: self.current_pos]

        self.current_pos += 1

        if self.total_diff > self.max_diff:
            self.max_diff = self.total_diff

        if self.trailing_end:
            self.min_diff = max(
                self.min_diff, self.total_diff - self.max_loss)

        if self.norm_obs:
            obs = (obs - self.obs_low) / \
                (self.obs_high - self.obs_low)

        return obs

    def get_norm_obs(self, obs):
        return (obs - self.observation_space.low) / \
            (self.observation_space.high - self.observation_space.low)

    def reset(self):
        # Trackers
        self.current_pos = self.obs_timesteps

        self.total_orders = 0
        self.total_reward = 0
        self.total_diff = 0
        self.total_bars = 0
        self.total_target = 0
        self.total_risk = 0
        self.max_diff = 0
        self.min_diff = -self.max_loss

        self.load_history()
        return self.get_obs()

    def advance_order(self, target, risk):
        tp = 0
        sl = 0
        open_price = 0

        open_bid = self.history[self.current_pos][0]
        open_ask = self.history[self.current_pos][1]

        if target > 0:
            # Long order opens by buying (ask) and closes by selling (bid)
            open_price = open_ask
            sl = round(open_bid - risk, 5)
            tp = round(open_ask + target, 5)

            def get_price_diff(index):
                ref_price = self.history[index][0]
                # nsl = round(ref_price - risk, 5)
                nsl = ref_price - risk
                done = ref_price >= tp or ref_price <= sl
                return done, max(nsl, sl), ref_price - open_price

        elif target < 0:
            # Short order opens by selling (bid) and closes by buying (ask)
            open_price = open_bid
            sl = round(open_ask - risk, 5)
            tp = round(open_bid + target, 5)

            def get_price_diff(index):
                ref_price = self.history[index][1]
                # nsl = round(ref_price + risk, 5)
                nsl = ref_price + risk

                done = ref_price <= tp or ref_price >= sl
                return done, min(nsl, sl), open_price - ref_price

        bars = 0
        price_diff = 0

        i = self.history_size

        for i in range(self.current_pos, self.history_size - 1):
            if self.history[i][2] == 1:
                bars += 1

            done, sl, price_diff = get_price_diff(i)

            if done or bars >= self.max_bars:
                break

        if self.single_position:
            self.current_pos = i

        self.total_diff += price_diff
        self.total_bars += bars

        if price_diff > 0:
            hold_modifier = pow(self.gamma, bars)
        else:
            hold_modifier = 1

        return (price_diff*hold_modifier), price_diff

    def step(self, action):
        target_percentage = action[0]
        risk_percentage = action[1]

        if (target_percentage != 0):
            self.total_orders += 1

            if target_percentage > 0:
                base_pts = self.min_target_pts
                risk_sign = 1
            elif target_percentage < 0:
                base_pts = -self.min_target_pts
                risk_sign = -1

            target_pts = base_pts + (target_percentage*self.target_pts_domain)
            risk_pts = self.min_target_pts + \
                (risk_percentage*self.target_pts_domain)

            self.total_target += abs(target_pts)
            self.total_risk += risk_pts

            real_target = target_pts*.00001
            real_risk = risk_pts*.00001*risk_sign

            reward, diff = self.advance_order(real_target, real_risk)

            # Risk reward modifier
            # if reward > 0:
            #     reward *= target_pts/risk_pts

        else:
            reward = 0
            diff = 0

        self.total_reward += reward

        return self.get_obs(), reward, self.check_done(), {"diff": diff, "total_diff": self.total_diff, "total_reward": self.total_reward, "total_orders": self.total_orders, "total_bars": self.total_bars}

    def check_done(self):
        if (self.current_pos >= self.history_size - 1) or (self.total_diff <= self.min_diff):
            if self.total_orders <= 0:
                self.total_orders = -1

            self.log("Episode End", "Reward: {:.5f}, Total Diff: {:.0f}, Max Diff: {:.0f}, Orders: {:d}, Mean Bars: {:.2f}, Mean Target: {:.2f}, Mean Risk {:.2f}".format(
                self.total_reward, self.total_diff*100000, self.max_diff*100000, self.total_orders, self.total_bars/self.total_orders, self.total_target/self.total_orders, self.total_risk/self.total_orders), 1)

            return True
        else:
            return False

    def log(self, prefix, msg, debug_level=2):
        if debug_level <= self.debug:
            print("[{}] {}".format(prefix, msg))

    def render(self, mode='human'):
        pass
