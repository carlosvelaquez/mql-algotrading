from prodigy import *
import random

env = Prodigy(mt5_path=None, single_step=True,
              csv_path="./prod_history.csv", debug_mode=3)

env.reset()

for _ in range(50):
    action = random.randint(0, 2)
    obs, reward, done, info = env.step(action)

    print("\nAction: {}, Reward: {}, Observation Shape: {}, Obs:".format(
        action, reward, obs.shape))
    print(obs)
    print("Done: {}\n".format(done))
