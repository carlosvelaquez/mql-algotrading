from prodigy import *

from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

mt5_path = r'C:\Users\imado\AppData\Roaming\MetaTrader 5 - EXNESS\terminal64.exe'

env = Prodigy(mt5_path, debug_mode=2)

nb_actions = env.action_space.n
input_shape = env.observation_space.shape

model = Sequential()

model.add(Reshape(input_shape, input_shape=(1,) + input_shape))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(nb_actions, activation='softmax'))

model.summary()
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

memory = SequentialMemory(limit=100000, window_length=1)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, policy=policy,
               nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000)

dqn.compile(optimizer=Adam(), metrics=['mae'])
dqn.fit(env, nb_steps=100000, visualize=False)
dqn.save_weights('dqn_{}_weights.h5f'.format("Prodigy"), overwrite=True)

dqn.test(env, 10000)
