import math
import random
import gym
from gym import spaces
import numpy as np

DRAW = 3


class TicTacToe(gym.Env):
    def __init__(self):
        self.board = np.zeros(shape=(3, 3), dtype=np.int)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(3, 3), dtype=np.int)
        self.action_space = spaces.Discrete(9)
        #self.play_against_human = play_against_human

        self.done = False
        self.symbol = 1
        self.opp_symbol = 2

    def get_observation(self):
        return np.copy(self.board)

    def board_is_full(self):
        if np.isin(0, self.board):
            return False
        else:
            return True

    def pos_is_available(self, pos):
        x = pos % 3
        y = int(pos/3)

        #print("Coords:", x, y)

        if self.board[x, y] == 0:
            return True
        else:
            return False

    def put_random(self):
        x = random.randint(0, 8)

        while not self.pos_is_available(x):
            x = random.randint(0, 8)

        return self.mark(x, self.opp_symbol)

    def reset(self):
        self.done = False
        self.board = np.zeros(shape=(3, 3), dtype=np.int)

        x = random.randint(0, 1)

        if x == 1:
            self.put_random()

        return self.get_observation()

    def mark(self, pos, sym):
        x = pos % 3
        y = int(pos/3)

        #print("Coords:", x, y)
        reward = -1

        if self.board[x, y] == 0:
            self.board[x, y] = sym
            valid = True
        else:
            valid = False

        if not valid:
            reward = -40
            self.done = True
        else:
            win = self.check_end()

            if win == self.symbol:
                reward = 20
                self.done = True
            elif win == self.opp_symbol:
                reward = -20
                self.done = True
            elif win == DRAW:
                reward = 10
                self.done = True

        return reward

    def check_end(self):
        for i in range(3):
            if self.board[i, 0] == self.board[i, 1] == self.board[i, 2]:
                sym = self.board[i, 0]
                if sym != 0:
                    return sym
            if self.board[0, i] == self.board[1, i] == self.board[2, i]:
                sym = self.board[0, i]
                if sym != 0:
                    return sym

        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2]:
            sym = self.board[0, 0]
            if sym != 0:
                return sym
        elif self.board[0, 2] == self.board[1, 1] == self.board[2, 0]:
            sym = self.board[0, 2]
            if sym != 0:
                return sym

        if not self.board_is_full():
            return 0
        else:
            return DRAW

    def step(self, act):
        reward = self.mark(act, self.symbol)

        if not self.done:
            reward = self.put_random()

        return self.get_observation(), reward, self.done, {}

    def render(self, mode='human'):
        print("---------------")
        for x in range(3):
            for y in range(3):
                print(self.board[y, x], end=" ")
            print("")
        print("---------------")
