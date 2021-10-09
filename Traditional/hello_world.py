import gym

env = gym.make('CartPole-v0')


for i_episode in range(20):
    obs = env.reset()
    print(env.action_space)
    print(env.observation_space)

    for t in range(1000):
        env.render()
        print(obs)

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after %i timesteps" % (t+1))
            break

env.close()
