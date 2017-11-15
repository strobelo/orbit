import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import pdb

tensors = {False: torch.FloatTensor, True: torch.cuda.FloatTensor}


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class QAgent(object):
    ''' 
    Use Q network to approximate sum of future reinforcements
    Action space must be Discrete
    '''

    def __init__(self, action_space, observation_space, qLayers=3, use_cuda=True, qlr=0.01, eld=0.99):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.tensorType = tensors[use_cuda]
        self.eld = eld
        self.epsilon = 1
        self.action_space = action_space
        self.qlr = qlr
        self.Q = Reltan(observation_space.sample().shape[0] + 1, h=2, o=1, layers=qLayers)
        self.prev = {'observation':None, 'reward':None, 'action':None}

    def act(self, observation, reward, maximize=True, epsilon=None, trainQIter=50):
        '''
        Assumes action space is spaces.Discrete
        maximize=True assumes we want to choose the action that
            maximizes the sum of future reinforcements
        '''
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() > epsilon:
            Qs = []
            for action in range(self.action_space.n):
                Qs.append(Q(oa2qin(observation, action)))
            action = np.argmax(np.array(Qs))
        else:
            action = self.action_space.sample()

        if trainQIter and self.prev['observation'] is not None:
            self.trainQ([self.prev['observation'],observation], [self.prev['action'],action], [self.prev['reward'],reward], trainQIter)

        self.prev['observation'], self.prev['action'], self.prev['reward'] = observation, action, reward
        return action

    def oa2qin(self, observations, actions):
        #return Variable(self.tensorType(np.hstack((observations, actions.reshape(actions.shape[0], 1)))))
        return Variable(torch.cat((observations,actions), dim=1))

    def trainQ(self, observations, actions, rewards, nIter):
        # observations = np.array(observations)
        # actions = np.array(actions)
        # rewards = np.array(rewards)
        observations = self.tensorType(observations)
        actions = self.tensorType(actions)
        rewards = self.tensorType(rewards)
        if observations.shape[0] is not rewards.shape[0]:
            raise ValueError(
                'shape[0]s do not match; rows of observations and rewards must align')
        qx = self.oa2qin(observations, actions)
        qprev = self.Q(qx)
        # add a zero at the end (will throw away this row after doing tensor
        # math)
        qcur = torch.cat((qprev[1:], self.tensorType([0])))
        qy = qcur.data + self.qlr * (rewards - qprev.data)
        qy = Variable(qy[:-1], requires_grad=False)  # remove that row at the end
        # train Q
        _train(self.Q, qx, qy, nIter, self.use_cuda)
        #self.epsilon = self.eld * observations.shape[0] * self.epsilon


def _train(model, x, y, nIter, use_cuda=True):
    # x = Variable(torch.FloatTensor(x))
    # y = Variable(torch.FloatTensor(y), requires_grad=False)
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

    def __init__(self, n, h, o, layers=3):
        super(Reltan, self).__init__()
        self.mods = torch.nn.ModuleList([])
        self.mods.append(torch.nn.Linear(n, h))
        for layer in range(layers):
            self.mods.append(torch.nn.ReLU())
            self.mods.append(torch.nn.Tanh())
        self.mods.append(torch.nn.Linear(h, o))

    def forward(self, x):
        # return self.fc(x) # it was just x there
        for module in self.mods:
            x = module(x)
        return x
