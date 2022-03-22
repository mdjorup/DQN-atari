import random
import time

from gym import envs
import gym

import numpy as np

import cv2

from PIL import Image

import tensorflow as tf
# seed = 42

#print(envs.registry.all())


env = gym.make("BreakoutNoFrameskip-v4", render_mode='rgb_array')

print(env.action_space)
print(env.action_space.n)

episodes = 1

# for episode in range(episodes):
#     state = env.reset()
#     done = False
#     score = 0
#     while not done:
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#         print(n_state.shape, type(n_state))
#         done=True

#         t0=time.time()
#         grayscale = tf.image.rgb_to_grayscale(n_state)
#         resize = tf.image.resize(grayscale, [110, 84])
#         cropped = tf.image.crop_to_bounding_box(resize, 17, 0, 84, 84)
#         print("Time:", time.time()-t0)

#         t0=time.time()
#         grayscale = tf.image.rgb_to_grayscale(n_state)
#         resize = tf.image.resize(grayscale, [110, 84])
#         cropped = tf.image.crop_to_bounding_box(resize, 17, 0, 84, 84)
#         print("Time:", time.time()-t0)

#         print(cropped.shape)

#         print('Got here successfully')
#         # cv2.imshow("Grayscale", grayscale.numpy())
#         # cv2.imshow("resized", resize.numpy())
#         #cv2.imshow("cropped", cropped.numpy())

#         key = cv2.waitKey(15000)#pauses for 15 seconds before fetching next image
#         if key == 27:#if ESC is pressed, exit loop
#             cv2.destroyAllWindows()
            


        
        
#         time.sleep(.1)
#     print("Episode: {}\t Score: {}".format(episode, score))
    
# env.close()
