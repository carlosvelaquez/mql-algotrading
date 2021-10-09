import gym
import tensorflow as tf
from tyro import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

env = Tyro('history.csv', history_timesteps=int(
    2*60*100), ticks_per_episode=int(2*60*60*8))
nb_actions = env.action_space.n
input_shape = env.observation_space.shape

model = Sequential()

model.add(Reshape(input_shape, input_shape=(1,) + input_shape))
model.add(LSTM(32, activation='relu',
               return_sequences=True, input_shape=input_shape))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(Dense(32, activation='relu'))
model.add(Flatten())
model.add(Dense(nb_actions, activation='linear'))

model.summary()
model.compile(optimizer=Adam(), loss='mse')

memory = SequentialMemory(limit=50000, window_length=1)

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(
), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)

dqn = DQNAgent(model=model, policy=policy, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=50)
dqn.compile(optimizer=Adam())

dqn.fit(env, nb_steps=int(2*60*60*24*150), visualize=False)
dqn.save_weights('dqn_{}_weights.h5f'.format("Tyro"), overwrite=True)
dqn.test(env, 10000)
