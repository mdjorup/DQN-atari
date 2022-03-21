import numpy as np
import tensorflow as tf

from preprocessing import phi

import replay_memory


#gamma
gamma = 0.99 #discount factor


#frame count
frame = 0
#max frames to train on
max_frames = 1000000
#how many frames to be random actions to build initial memory
random_frames = 10000 
#how many frames to decay epsilon 
e_greedy_frames = 200000
#how often to update model, memory added on every frame
update_interval = 4

#Frequency in which random actions are taken
epsilon = 1
#lowest possible frequency for random actions
min_epsilon = 0.1
#epsilon decrement on each frame
epsilon_decrement = (epsilon - min_epsilon) / e_greedy_frames



batch_sample_size = 32


memory = replay_memory.ReplayMemory(10000)


print(memory.max_capacity)
# macros



#logic 
# We have initial image x
# We need to turn image x into s0 in initialization
# We get new image
    # Need to process image
    # Add it to s
    #remove first x in s
    #now we have s1



