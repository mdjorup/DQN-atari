import numpy as np
import tensorflow as tf
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import os
import gym
import replay_memory
import state
from model import build_model 



checkpoint_path = "checkpoints/cp-{epoch:07d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

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
                    state = np.array([state_processor.get_state()])
                    preds = self.target_model.predict(state)
                    action = np.argmax(preds)
                

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
                    
                    #sample mini batch of transitions
                    initial_state_samples, action_samples, reward_samples, next_state_samples, done_samples = self.memory.sample(self.batch_sample_size)

                    #This stores the expected rewards for each state transition - 
                    #   = r if transition is terminal
                    #   = r + gQ*(next state, a', theta)  --> basically passing the next state through the network and getting the max action value
                    # should be an array of action values

                    #model target is what we do the predictions on, model is what we update on
                    # every n frames we set model target to whatever model is.
                    
                    # 1. getting y

                    #pass next state through the network, get an action value for each action
                    next_predictions = self.target_model.predict(next_state_samples)
                    #gets the max action value 
                    max_q = np.reshape(tf.math.reduce_max(next_predictions, axis=1), (-1,))
                    # if done is True, then make the q value equal to 0. 
                    max_q = max_q * (1 - done_samples)

                    y = reward_samples + self.gamma * max_q

                    # mask to indicate correct indexes to collect the action values
                    masks = tf.one_hot(action_samples, 4)

                    
                    with tf.GradientTape() as tape:

                        #train the model on the initial states
                        prev_predictions = self.model(initial_state_samples)

                        #get all of the action values for the actions that were taken
                        prev_actions = tf.reduce_sum(tf.multiply(prev_predictions, masks), axis=1)

                        # compute the loss
                        current_loss = self.loss_function(y, prev_actions) 

                    grads = tape.gradient(current_loss, self.model.trainable_variables)
                    
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if done or episode_frames >= self.max_frames_per_episode:
                    break
            
            #whenever an episode is completed, update the target model
            self.target_model.set_weights(self.model.get_weights())


            if self.episode % self.episodes_per_epoch == 0:
                self.target_model.save_weights(checkpoint_path.format(epoch=self.episode))
                pass

            self.episode_rewards.append(episode_reward)
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)

            avg_episode_reward = np.mean(self.episode_rewards)

            #do printing for model monitoring
            print("Episode: {} \tFrame: {} \tAvg Reward: {} \tEpsilon: {}".format(self.episode, self.frame, avg_episode_reward, self.epsilon), end='\r')

            #break training conditions

            if avg_episode_reward > 40:
                #save model target weights
                self.target_model.save_weights(checkpoint_path.format(epoch=self.episode))
                
            
            if self.frame >= self.max_frames:
                #save target model weights
                self.target_model.save_weights(checkpoint_path.format(epoch=self.episode))
                


               



