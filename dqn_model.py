import numpy as np
import tensorflow as tf
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import os
import gym
import replay_memory
import state
from model import build_model 



class DQN_Model:

    def __init__(self, gamma=0.99, memory_size=25000, max_frames=1000000, random_frames=15000, e_greedy_frames=200000, max_frames_per_episode=10000, max_epsilon=1, min_epsilon=0.1, batch_sample_size=32):
        # immutable
        self.gamma = gamma
        self.memory_size = memory_size
        self.max_frames = max_frames
        self.random_frames = random_frames
        self.e_greedy_frames = e_greedy_frames
        self.max_frames_per_episode = max_frames_per_episode
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.batch_sample_size = batch_sample_size
        self.optimizer = Adam(learning_rate=0.00025, clipnorm=1.0)
        self.loss_function = MeanSquaredError()
        self.episodes_per_epoch = 20
        self.update_interval = 4
        

        #env
        self.env = gym.make("BreakoutNoFrameskip-v4", render_mode="rbg_array")
        self.num_actions = self.env.action_space.n

        #mutable
        self.model = build_model((84, 84, 4), self.num_actions)
        self.target_model = build_model((84, 84, 4), self.num_actions)
        self.memory = replay_memory.ReplayMemory(self.memory_size)
        self.frame = 0
        self.epsilon = self.max_epsilon
        self.episode = 0
        self.episode_rewards = []


        #checkpoint path




    def load_from_checkpoint(self):
        #sets all variables that are saved in the checkpoint
        # this should load the latest checkpoint and update the 
        # mutable variables
        pass

        



    def train(self, checkpoints):
        #this is training 
        while checkpoints > 0 :

            episode_reward = 0
            episode_frames = 0

            self.episode += 1

            initial_image = self.env.reset()

            state_processor = state.StateProcessor(initial_image, 4)

            while True:

                #action selection
                if self.frame < self.random_frames or np.random.uniform() < self.epsilon:
                    action = self.env.action_space.sample()
                #condition to take action selected by model
                else:
                    #TODO: need to take action with highest Q
                    action = self.env.action_space.sample()
                

                #execute action in environment
                prev_state = state_processor.get_state()
                rewards = 0
                for _ in range(self.update_interval):
                    #take action and update environment
                    new_frame, reward, done, info = self.env.step(action)
                    #accumilate rewards
                    rewards += reward
                    #update total frame count
                    self.frame += 1
                    episode_frames += 1
                    if done:
                        break
                
                #scale rewards
                if rewards > 0:
                    rewards = 1
                elif rewards < 0:
                    rewards = -1
                else:
                    rewards = 0

                #accumilate episode_reward
                episode_reward += rewards

                #insert latest frame into sp
                state_processor.insert_frame(new_frame)

                #get updated state
                new_state = state_processor.get_state()

                #insert 
                self.memory.insert(prev_state, action, reward, new_state, done)

                if self.frame >= self.random_frames:
                    self.epsilon = max(self.max_epsilon - (self.frame / self.e_greedy_frames)*(self.max_epsilon - self.min_epsilon), self.min_epsilon)
                    



                



