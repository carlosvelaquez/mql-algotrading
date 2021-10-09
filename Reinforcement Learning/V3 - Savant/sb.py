from stable_baselines import PPO2, A2C
from savant import *

env = Savant(obs_window_timesteps=1200, debug_mode=1)

# For recurrent policies, with PPO2, the number of environments run in parallel
# should be a multiple of nminibatches.
model = PPO2('MlpPolicy', env, nminibatches=1, verbose=2)
# model = A2C('MlpLstmPolicy', env, verbose=2)
model.learn(100000)

obs = env.reset()
# Passing state=None to the predict function means
# it is the initial state
state = None

for _ in range(1000):
    # We need to pass the previous state and a mask for recurrent policies
    # to reset lstm state when a new episode begin
    action, state = model.predict(obs, state=state)
    obs, reward, done, _ = env.step(action)
    # Note: with VecEnv, env.reset() is automatically called

print("done")
