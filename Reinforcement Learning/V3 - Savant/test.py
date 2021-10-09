from savant import *
import random

env = Savant(obs_window_timesteps=5, debug_mode=2)

done = False

while not done:
    obs, reward, done, info = env.step(random.randint(0, 4))

    print("-------------")
    print(obs.shape)
    for x in obs:
        print(x)
    print("-------------")

# env.reset()

# done = False
# while not done:
#     obs, reward, done, info = env.step(2)

print("done")
