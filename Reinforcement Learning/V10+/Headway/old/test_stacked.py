from headway_stacked import *

env = Headway(gamma=0.965)
obs = env.reset()

print(obs)

for i in range(100000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    print(action)
    print(obs.shape)
    print(obs)
    print(done)

    if done:
        env.reset()
