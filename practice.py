import gym
import random
import time
# seed = 42


# print(env.action_space)

episodes = 1


env = gym.make('BreakoutNoFrameskip-v0')

print(env.action_space)

for episode in range(episodes):
  state = env.reset()
  done = False
  score = 0

  while not done:
    env.render()
    action = random.choice([0, 1, 2, 3])
    n_state, reward, done, info  = env.step(action)
    score += reward
    done = True
    time.sleep(0.5)
  print("Episode: {}\t Score: {}".format(episode, score))
  print(type(n_state))
env.close()