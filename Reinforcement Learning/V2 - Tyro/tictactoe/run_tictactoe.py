import gym
from tictactoe import *

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Flatten, Conv2D, BatchNormalization
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

env = TicTacToe()
env.reset()
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) +
                  env.observation_space.shape))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer=Adam())

memory = SequentialMemory(limit=1000, window_length=1)
policy = BoltzmannQPolicy()
#policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)

dqn = DQNAgent(model=model, policy=policy, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=1000)
dqn.compile(optimizer=Adam())

dqn.fit(env, nb_steps=20000, visualize=False)
dqn.test(env, 25)
