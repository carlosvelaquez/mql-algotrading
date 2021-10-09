from stable_baselines import PPO2, A2C, DQN
from stable_baselines.common.evaluation import evaluate_policy
from prodigy import *

# mt5_path = r'C:\Users\imado\AppData\Roaming\MetaTrader 5 - EXNESS\terminal64.exe'
# env = Prodigy(mt5_path, advance_days=5, debug_mode=2)
env = Prodigy(mt5_path=None, csv_path="./prod_history.csv", debug_mode=2)

# Instantiate the agent
model = A2C('MlpLstmPolicy', env, verbose=2)
# Train the agent
model.learn(total_timesteps=int(25000000))
# Save the agent
model.save("prodigy_model")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = A2C.load("prodigy_model")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=5)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

print("done")
