#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from dqn_model import DQN
import numpy as np
from agent import Agent
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple, deque
from collections import deque
import matplotlib.pyplot as plt
import copy

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

EPISODES = 40000
LEARNING_RATE = 1.5e-4  # alpha
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 400     ### These hyperparameters are accoriding to the slides 
EPSILON = 1.0
EPSILON_END = 0.025
FINAL_EXPL_FRAME = 1000000
TARGET_UPDATE_FREQUENCY = 1000
SAVE_MODEL_AFTER = 5000
DECAY_EPSILON_AFTER = 10000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################### https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html #############

reward_buffer = deque([0.0], maxlen = 100)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state','reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.mem = deque([],maxlen = capacity)

    def push(self, *args):
        self.mem.append(Transition(*args))

    def sample(self, batch_size):
        tuples=random.sample(self.mem,batch_size)
        return tuples

    def __len__(self):
        return len(self.mem)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.en = env
        self.count = self.en.action_space.n
        input_photos = 4
        self.QNET = DQN(input_photos,self.count).to(device)
        self.Target_QNET = copy.deepcopy(self.QNET).to(device)
        self.buffer_ = ReplayMemory(BUFFER_SIZE)
        self.optimizer = optim.Adam(self.QNET.parameters(), lr = LEARNING_RATE)
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #

            self.QNET=DQN(input_photos,self.count)
            self.QNET.load_state_dict(torch.load('./DQN.pth',map_location=device))
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        observation = np.array(observation,dtype = np.float32)/255
        observation = observation.transpose(2,0,1)
        observation = np.expand_dims(observation,0)
        observation = torch.from_numpy(observation)
        logs = self.QNET(observation.to(device))
        maxActions = torch.argmax(logs)      
        ###########################
        return maxActions.detach().item()
    
        

    def train(self,episodes=EPISODES):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        epi_reward = []
        count = 0

        for epis in range(episodes):
            print(f'The current episode running is: {epis}')
            episode_reward = 0
            current_state = self.env.reset()
            done = False
            while not done:
                if epis > DECAY_EPSILON_AFTER:
                    epsilon = np.interp(count, [0, FINAL_EXPL_FRAME], [EPSILON, EPSILON_END])
                else:
                    epsilon = EPSILON
                action = self.make_action(current_state)
                pro = np.ones(self.count) * epsilon / self.count  
                pro[action] += 1 - epsilon 
                final_action = np.random.choice(np.arange(self.count), p = pro) 
                next_state,reward,done,_,_ = self.env.step(final_action)
                episode_reward += reward
                h = np.array(current_state,dtype=np.float32)/255
                h = h.transpose(2, 0, 1)
                h = np.expand_dims(h, 0)
                current_state_tensor=torch.from_numpy(h)
                h = np.array(next_state,dtype=np.float32)/255
                h = h.transpose(2, 0, 1)
                h = np.expand_dims(h, 0)
                ns_tensor = torch.from_numpy(h)
                action_t = torch.tensor([action], device=device)
                reward_t = torch.tensor([reward], device=device)
                self.buffer_.push(current_state_tensor, action_t,ns_tensor,reward_t)
                current_state = next_state
                self.optimize()
                if done:
                    reward_buffer.append(episode_reward)
                    break
                count += 1
            if epis % TARGET_UPDATE_FREQUENCY == 0:
                self.Target_QNET.load_state_dict(self.QNET.state_dict())
            if epis % SAVE_MODEL_AFTER == 0:
                torch.save(self.QNET.state_dict(), "DQN.pth")
            epi_reward.append(episode_reward)
        torch.save(self.QNET.state_dict(), "DQN.pth")
        ax = plt.figure()
        plt.plot(range(len(epi_reward)),epi_reward)
        plt.title('Reward VS Epi Number')
        plt.xlabel('Epi Number')
        plt.ylabel('Reward')
        plt.savefig('image.png')
        print("DOneeee!")
        ###########################


    def optimize(self):
        if len(self.buffer_) < BATCH_SIZE:
            return
        transisions = self.buffer_.sample(BATCH_SIZE)
        batch = Transition(*zip(*transisions))
        temp_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        temp_nextStates = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward)
        temp = self.QNET(state_batch) 
        S_A_VALUES = temp[torch.arange(temp.size(0)), action_batch]
        N_S_values = torch.zeros(BATCH_SIZE, device = device)
        N_S_values[temp_mask] = self.Target_QNET(temp_nextStates.float()).max(1)[0].detach()
        expected_S_A_VALUES = (N_S_values * GAMMA) + reward_batch
        conditions = nn.SmoothL1Loss()
        loss = conditions(S_A_VALUES, expected_S_A_VALUES)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.QNET.parameters():
            param.grad.data.clamp_(-1, 1)  
        self.optimizer.step()