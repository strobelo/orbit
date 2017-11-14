import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

tensors = {False:torch.nn.FloatTensor, True:torch.cuda.FloatTensor}

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class QPAgent(object):
    def __init__(self, action_space, initial_state,qLayers=3, pLayers=3, use_cuda=True, qlr=0.01, plr=0.01):
        self.action_space = action_space
        self.qlr, self.plr = qlr, plr
        self.Q = Reltan(initial_state.shape[1]+1, h=10, o=1, layers=qLayers)
        actionShape = 1
        try:
            actionShape = action_space.sample().shape[0]
        except AttributeError as e:
            pass
        self.P = Reltan(initial_state.shape[1], h=10, o=actionShape, layers=pLayers)

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def oa2qin(observations, actions):
        return tensors[use_cuda](np.hstack(observations,actions.reshape(actions.shape[0],1)))

    def trainQ(self, observations, actions, rewards):
    	if observations.shape[0] is not rewards.shape[0]:
    		raise ValueError('shape[0]s do not match; rows of observations and rewards must align')
        qx = oa2qin(observations,actions)
        qpred = Q(qx)[1:]
        qy = qpred + self.qlr * (rewards + None ) # this is where you left off
		#train Q
		#train R
		

def _train(model, x, y, nIter, use_cuda=True):
    x = Variable(torch.FloatTensor(x))
    y = Variable(torch.FloatTensor(y), requires_grad=False)
    if torch.cuda.is_available() and use_cuda:
        model = model.cuda()
        x = x.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    lossTrace = []
    for t in range(nIter):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        lossTrace.append(loss.data[0])
        #print(t, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return lossTrace

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