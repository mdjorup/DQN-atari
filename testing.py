from model import build_model 
import state
import gym
import numpy as np
import time
import os
import tensorflow as tf


checkpoint_path = "checkpoints/training_4/cp-{epoch:07d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")


best_model = build_model((84, 84, 4), 4)


latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
best_model.load_weights(latest)


episodes = 1

for episode in range(episodes):
    initial_frame = env.reset()

    sp = state.StateProcessor(initial_frame)
    done = False
    score = 0
    while not done:

        if np.random.uniform() < 0.05:
            action = np.random.choice(4)
        else:
            cur_state = np.array([sp.get_state()])
            preds = best_model.predict(cur_state)
            action = np.argmax(preds)

        for _ in range(4):
            n_frame, reward, done, info = env.step(action)
            if reward > 0:
                print("Score Change")
            score += reward
            if done:
                break
        sp.insert_frame(n_frame)

        
    print("Episode Score: {}".format(score))

env.close()



