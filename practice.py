import random
import time

from gym import envs
import gym
# seed = 42

#print(envs.registry.all())


env = gym.make("BreakoutNoFrameskip-v4", render_mode='human')

print(env.action_space)

episodes = 1

for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
        time.sleep(.1)
    print("Episode: {}\t Score: {}".format(episode, score))
    
env.close()
