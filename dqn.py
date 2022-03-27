from turtle import ycor
import numpy as np
import tensorflow as tf
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

import gym

import replay_memory
import state

from model import build_model 


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

optimizer = Adam(learning_rate=0.00025, clipnorm=1.0)

loss_function = MeanSquaredError()

#1. Define environment
env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
num_actions = env.action_space.n

#2. Initialize replay memory
memory = replay_memory.ReplayMemory(40000)


#3. Create neural nework 

model = build_model((84, 84, 4), num_actions)

target_model = build_model((84, 84, 4), num_actions)


episode = 0
episode_rewards = []

while True:
    
    episode_reward = 0
    episode_frames = 0

    episode += 1

    initial_image = env.reset()

    state_processor = state.StateProcessor(initial_image, 4)



    while True: #this is each episode
        
        ##########################################################################

        #action selection: only happens on every fourth frame or if we get to a new environment
        if frame < random_frames or np.random.uniform() < epsilon:
            action = env.action_space.sample()
        # condition to take action selected by the model
        else:
            #TODO: take action with the max Q value given the state_processor state
            action = env.action_space.sample()

        ##########################################################################

        #Execute action in the environment

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
            episode_frames += 1
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

        episode_reward += reward

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

        #############################################################################


        # Network updates if we are past the random state
        if frame >= random_frames:
            #update epsilon
            epsilon = max(max_epsilon - (frame / e_greedy_frames)*(max_epsilon-min_epsilon), min_epsilon)

            # sample a mini batch of transitions - all are in the form of a list
            initial_state_samples, action_samples, reward_samples, next_state_samples, done_samples = memory.sample(batch_sample_size)

            #This stores the expected rewards for each state transition - 
            #   = r if transition is terminal
            #   = r + gQ*(next state, a', theta)  --> basically passing the next state through the network and getting the max action value
            # should be an array of action values

            #model target is what we do the predictions on, model is what we update on
            # every n frames we set model target to whatever model is.
            
            # 1. getting y

            #pass next state through the network, get an action value for each action
            next_predictions = target_model.predict(next_state_samples)
            #gets the max action value 
            max_q = np.reshape(tf.math.reduce_max(next_predictions, axis=1), (-1,))
            # if done is True, then make the q value equal to 0. 
            max_q = max_q * (1 - done_samples)

            y = reward_samples + gamma * max_q

            
            # mask to indicate correct indexes to collect the action values
            masks = tf.one_hot(action_samples, 4)

            
            with tf.GradientTape() as tape:

                #train the model on the initial states
                prev_predictions = model(initial_state_samples)

                #get all of the action values for the actions that were taken
                prev_actions = tf.reduce_sum(tf.multiply(prev_predictions, masks), axis=1)

                # compute the loss
                current_loss = loss_function(y, prev_actions) 

            grads = tape.gradient(current_loss, model.trainable_variables)
            
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            quit()
                    
    
        #break conditions
        if done or episode_frames >= max_frames_per_episode:
            break

    #whenever an episode is completed, update the target model
    target_model.set_weights(model.get_weights())
    

    episode_rewards.append(episode_reward)
    if len(episode_rewards) > 100:
        episode_rewards.pop(0)

    avg_episode_reward = np.mean(episode_rewards)

    print("Got to episode {}, frames: {}, Average Episode Reward: {}".format(episode, frame, avg_episode_reward), end='\r')

    if avg_episode_reward > 40:
        break

    if frame >= max_frames:
        break

print("Average episode reward after training: {}".format(avg_episode_reward))

    




