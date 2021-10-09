import zmq
import warnings
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from apex import *

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# ZMQ Socket
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:5555")

# Apex Agent


def make_env():
    return Apex(
        max_account_loss=.05, min_target_points=25, max_target_points=250, max_opt_bars=60, min_risk_ratio=.5, debug_mode=0)


env = make_env()

vec_env = VecNormalize.load(
    "apex_vecenv", make_vec_env(make_env, n_envs=16))

vec_env.training = False
vec_env.norm_reward = False

model = PPO2.load("apex_model")
num_envs = vec_env.num_envs
num_features = 19

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
        norm_obs = vec_env.normalize_obs(obs)
        multi_obs = np.tile(norm_obs, [num_envs, 1])

        action, state = model.predict(
            multi_obs, state=state, mask=None, deterministic=True)

        target, risk = env.convert_action(action[0])

        if action[0][0] < 0:
            pass

        socket.send_string(str(target) + ',' + str(risk))

except (KeyboardInterrupt, SystemExit):
    print("Interrupted. Server shutting down...")
finally:
    print("Server shutting down...")
    socket.close()
