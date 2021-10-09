from savant import *

from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

env = Savant(obs_window_timesteps=1200, test_days=1, debug_mode=1)

nb_actions = env.action_space.n
input_shape = env.observation_space.shape

model = Sequential()

model.add(Reshape(input_shape, input_shape=(1,) + input_shape))
model.add(LSTM(32, activation='relu',
               return_sequences=True, input_shape=input_shape))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(Dense(32, activation='relu'))
model.add(Flatten())
model.add(Dense(nb_actions, activation='softmax'))

model.summary()
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

memory = SequentialMemory(limit=50000, window_length=1)

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(
), attr='eps', value_max=1., value_min=.1, value_test=0., nb_steps=5000)

dqn = DQNAgent(model=model, policy=policy, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=1000)
dqn.compile(optimizer=Adam())

dqn.fit(env, nb_steps=int(1000000), visualize=False)
dqn.save_weights('dqn_{}_weights.h5f'.format("Savant"), overwrite=True)
dqn.test(env, 10000)
