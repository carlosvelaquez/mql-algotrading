import zmq
import warnings
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from headway import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# ZMQ Socket
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:5555")

# Headway Agent


def make_env():
    return Headway(gamma=.965)


env = make_env()

model = PPO2.load("headway_model")
num_envs = 16

state = None


try:
    print("Server up. Waiting for MT5 interface...")

    while True:
        msg = socket.recv().decode("utf-8")

        if msg == "connect":
            print("Received connect message:", msg)
            socket.send_string("Server connected.")
            print("Connected to MT5 interface successfully.")
            state = None
            continue

        obs = np.array(msg.split(','), dtype='f')
        obs = np.tile(env.get_norm_obs(obs), [num_envs, 1])

        action, state = model.predict(
            obs, state=state, mask=None, deterministic=True)

        # if action[0][0] < 0 or action[0][0] > .3:
        #     target = 0
        # else:
        target, risk = env.convert_action(action[0])

        socket.send_string(str(target) + ',' + str(risk))

except (KeyboardInterrupt, SystemExit):
    print("Interrupted. Server shutting down...")
finally:
    print("Server shutting down...")
    socket.close()
