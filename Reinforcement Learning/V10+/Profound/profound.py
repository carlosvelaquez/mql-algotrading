import gym

class Profound(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)


        