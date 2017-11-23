import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch.nn as nn
import random
import torch.nn.functional as F

import pdb

tensors = {False: torch.FloatTensor, True: torch.cuda.FloatTensor}


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

def flat(tensor):
        return tensor.view(tensor.size()[0])

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.o = []
        self.r = []
        self.a = []

    def add(self, observations, rewards, actions):
        self.o += observations
        self.r += rewards
        self.a += actions
        if len(self.o) > self.capacity:
            self.o.pop(0)
            self.r.pop(0)
            self.a.pop(0)

    def sample(self, size):
        o,r,a = [],[],[]
        for i in range(size):
            s = random.randint(0,len(self.o))
            o += self.o[s]
            r += self.r[s]
            a += self.a[s]
        return o,r,a

    def isEmpty(self):
        return len(self.o) > 0

class QAgent(object):
    ''' 
    Use Q network to approximate sum of future reinforcements
    Action space must be Discrete
    '''

    def __init__(self, action_space, observation_space, qLayers=10, qhidden=30, use_cuda=True, qlr=0.1, eld=0.99, gammaF=lambda x:0.1):
        self.replay = ExperienceReplay(10000)
        self.gamma = 0.1
        self.gammaF = gammaF
        self.use_cuda = (use_cuda and torch.cuda.is_available())
        self.tensorType = tensors[self.use_cuda]
        self.eld = eld
        self.epsilon = 1
        self.action_space = action_space
        self.qlr = qlr
        #self.Q = Reltan(observation_space.sample().shape[0] + 1, h=qhidden, o=1, layers=qLayers)
        nin = observation_space.sample().shape[0] + 1
        self.Q = nn.Sequential(nn.Linear(nin,qhidden), nn.ReLU6(), nn.ReLU6(), nn.ReLU6(), nn.Tanh(), nn.Tanh(), nn.Linear(qhidden,2), nn.Linear(2,1))
        if self.use_cuda:
            self.Q = self.Q.cuda()
        self.prev = {'observation':None, 'reward':None, 'action':None}
        self.lossTrace = np.array([])
        self.epoch=0

    def _amaxQ(self, observation, ret='a'):
        Qs = []
        for action in range(self.action_space.n):
            Qs.append(self.Q(self.oa2qin(observation,action)).data[0])
        Qs = np.array(Qs)
        if ret is 'a':
            return int(np.argmax(Qs))
        elif ret is 'q':
            return float(np.max(Qs))
        else:
            raise ValueError('Parameter "ret" must be either "a" or "q"')


    def act(self, observation, reward, maximize=True, epsilon=None, trainQIter=None):
        '''
        Assumes action space is spaces.Discrete
        maximize=True assumes we want to choose the action that
            maximizes the sum of future reinforcements
        '''
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() > epsilon:
            action = self._amaxQ(observation, ret='a')
        else:
            action = self.action_space.sample()

        if trainQIter and self.prev['observation'] is not None:
            self.trainQ([self.prev['observation'],observation], [self.prev['action'],action], [self.prev['reward'],reward], trainQIter)

        self.prev['observation'], self.prev['action'], self.prev['reward'] = observation, action, reward
        self.epsilon = max(self.eld * self.epsilon, 0.1)
        return action

    def oa2qin(self, observations, actions):
        #return Variable(self.tensorType(np.hstack((observations, actions.reshape(actions.shape[0], 1)))))
        try:
            return Variable(torch.cat((observations,actions), dim=1))
        except TypeError as e:
            return Variable(torch.cat((self.tensorType(observations),self.tensorType([actions]))))

    def trainQ(self, observations, actions, rewards, nIter):
        # observations = np.array(observations)
        # actions = np.array(actions)
        # rewards = np.array(rewards)
        self.replay.add(observations,actions,rewards)
        observations = self.tensorType(observations)
        actions = self.tensorType(actions)
        rewards = self.tensorType(rewards)
        if observations.shape[0] is not rewards.shape[0]:
            raise ValueError(
                'shape[0]s do not match; rows of observations and rewards must align')
        qx = self.oa2qin(observations, actions)
        qs = self.Q(qx).data
        qs = flat(qs)
        # qcur = torch.cat((qprev[1:], self.tensorType([0])))
        # qcur = qcur.data.view(qcur.size()[0])
        #qy = qcur.data.view(qcur.size()[0]) + self.qlr * (rewards - qprev.data.view(qprev.size()[0])) # SARSA
        #pdb.set_trace()
        qopt = self.tensorType(np.max(np.array([[self.Q(self.oa2qin(o,a)).data[0] for a in range(self.action_space.n)] for o in observations]), axis=1))
        #pdb.set_trace()
        qy = (1-self.qlr) * qs + self.qlr * (rewards + self.gamma * qopt)  # QLearning
        #pdb.set_trace()
        qy = Variable(qy, requires_grad=False)
        # train Q
        loss = _train(self.Q, qx, qy, nIter, self.use_cuda)
        self.lossTrace = np.append(self.lossTrace, loss)
        self.epoch +=1
        self.gamma = self.gammaF(self.gamma)
        #self.epsilon = self.eld * observations.shape[0] * self.epsilon


def _train(model, x, y, nIter, use_cuda=True):
    # x = Variable(torch.FloatTensor(x))
    # y = Variable(torch.FloatTensor(y), requires_grad=False)
    # if torch.cuda.is_available() and use_cuda:
    #     model = model.cuda()
    #     x = x.type(torch.cuda.FloatTensor)
    #     y = y.type(torch.cuda.FloatTensor)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    #optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
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
            self.mods.append(torch.nn.Tanh())
        self.mods.append(torch.nn.Linear(h, o))

    def forward(self, x):
        # return self.fc(x) # it was just x there
        for module in self.mods:
            module(x)
        return x
