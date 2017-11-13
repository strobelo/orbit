import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class QPAgent(object):
    def __init__(self, action_space, initial_state):
        self.action_space = action_space
        self.Q = Reltan(initial_state.shape[1]+1, h=10, o=1, layers=3)
        self.P = Reltan(initial_state.shape[1], h=10, o=1, layers=3)

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def train(self, observations, rewards):
    	if observations.shape[0] is not rewards.shape[0]:
    		raise ValueError('shape[0]s do not match; rows of observations and rewards must align')
    	for o in observations:
    		#train Q
    		#train R
    		pass

class Reltan(torch.nn.Module):
    def __init__(self,n,h,o,layers=3):
        super(Model, self).__init__()
        self.mods = torch.nn.ModuleList([])
        self.mods.append(torch.nn.Linear(n, h))
        for layer in range(layers):
	        self.mods.append(torch.nn.ReLU())
	        self.mods.append(torch.nn.Tanh())
        self.mods.append(torch.nn.Linear(h,o))
        
    def forward(self, x):
        #return self.fc(x) # it was just x there
        for module in self.mods:
            x = module(x)
        return x