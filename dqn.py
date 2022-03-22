from sre_parse import State
import numpy as np
import tensorflow as tf

import gym

import replay_memory
import state


#gamma
gamma = 0.99 #discount factor

#memory
memory_size = 10000

#frame count
frame = 0
#max frames to train on
max_frames = 1000000
#how many frames to be random actions to build initial memory
random_frames = 10000 
#how many frames to decay epsilon 
e_greedy_frames = 200000
#max frames per episode
max_frames_per_episode = 10000
#how often to update model, memory added on every frame
update_interval = 4

#highest possible frequency for random actions
max_epsilon = 1
#lowest possible frequency for random actions
min_epsilon = 0.1
#Frequency in which random actions are taken
epsilon = max_epsilon

batch_sample_size = 32

#1. Define environment
env = gym.make("BreakoutNoFrameskip-v4")
num_actions = env.action_space.n

#2. Initialize replay memory
memory = replay_memory.ReplayMemory(40000)


#3. Create neural nework 


while True:
    episode_rewards = []
    episode_reward = 0
    episode_frames = 0

    initial_image = env.reset()

    state_processor = state.StateProcessor(initial_image, 4)



    while True: #this is each episode
        

        #action selection: only happens on every fourth frame or if we get to a new environment
        if frame < random_frames or np.random.uniform() < epsilon:
            action = env.action_space.sample()
        # condition to take action selected by the model
        else:
            #TODO: take action with the max Q value given the state_processor state
            action = env.action_space.sample()

        
        # Get the current state of the environment
        prev_state = state_processor.get_state()
        rewards = 0
        # update the environment 4 times, taking the same action for 4 frames
        for _ in range(update_interval):
            # update environment
            new_frame, reward, done, info = env.step(action)
            #accumilate rewards
            rewards += reward
            #update total frame count
            frame += 1
            # if done, can't go more so stop iterating
            if done:
                break
        
        #need to scale rewards to between 0 and 1
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        else:
            reward = 0

        #insert a frame into the state processor - this frame is 4 frames in the future from the previous frame
        state_processor.insert_frame(new_frame)

        #get the new state from the state processor
        new_state = state_processor.get_state()

        # now we have an initial state, 
        # an action, 
        # the reward for taking that action
        # whether or not the episode terminated
        # a new state
        # time to insert this observation into our memory
        memory.insert(prev_state, action, reward, new_state, done)



        # Network updates if we are past the random state
        if frame >= random_frames:
            initial_state_samples, action_samples, reward_samples, next_state_samples, done_samples = memory.sample(batch_sample_size)





        # re define epsilon
        epsilon = max_epsilon - (frame / e_greedy_frames)*(max_epsilon-min_epsilon)
    
        #break conditions
        if done or episode_frames >= max_frames_per_episode:
            break


    #exit big loop conditions
    if frame >= max_frames:
        break

    episode_rewards.append(episode_reward)
    if len(episode_rewards) > 100:
        episode_rewards.pop(0)

    if sum(episode_rewards) / len(episode_rewards) > 40:
        break

        
    #exit condition



# macros



#logic 
# We have initial image x
# We need to turn image x into s0 in initialization
# We get new image
    # Need to process image
    # Add it to s
    #remove first x in s
    #now we have s1



