#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.c1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.c3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.l1 = nn.Linear(7*7*64,512)   
        self.l2 = nn.Linear(512,256)
        """ 000"""
        self.l3 = nn.Linear(256,num_actions)
        self.ReLu = nn.ReLU()


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        out = self.ReLu(self.c1(x))
        out = self.ReLu(self.c2(out))
        out = self.ReLu(self.c3(out))
        out = torch.flatten(out, 1) 
        out = self.ReLu(self.l1(out))
        out = self.ReLu(self.l2(out))
        out = self.l3(out)

        ###########################
        return out