from stable_baselines import PPO2, A2C, DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv
from prodigy import *

mt5_path = r'C:\Users\imado\AppData\Roaming\MetaTrader 5 - EXNESS\terminal64.exe'
env = Prodigy(mt5_path, advance_days=5, debug_mode=2)

model = A2C.load("prodigy_model_res")
model.set_env(DummyVecEnv([lambda: env]))

# Train the agent
model.learn(total_timesteps=int(2500000))
model.save("prodigy_model_res")

print("done")
