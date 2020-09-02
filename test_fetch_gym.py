import gym
env = gym.make('CartPole-v0')
# env = gym.make("FetchPushCustom-v1", n_substeps=20)
# env = gym.make("FetchPickAndPlace-v1")
#env = gym.make("FetchReach-v1")
# env = gym.make("Ant-v2")
#env = SawyerReachPushPickPlaceEnv()
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
