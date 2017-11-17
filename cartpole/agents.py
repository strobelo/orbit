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

def flat(tensor):
        return tensor.view(tensor.size()[0])

class QAgent(object):
    ''' 
    Use Q network to approximate sum of future reinforcements
    Action space must be Discrete
    '''

    def __init__(self, action_space, observation_space, qLayers=5, qhidden=10, use_cuda=True, qlr=0.01, eld=0.99, gammaF=lambda x:0.3):
        self.gamma = gammaF
        self.use_cuda = (use_cuda and torch.cuda.is_available())
        self.tensorType = tensors[self.use_cuda]
        self.eld = eld
        self.epsilon = 1
        self.action_space = action_space
        self.qlr = qlr
        self.Q = Reltan(observation_space.sample().shape[0] + 1, h=qhidden, o=1, layers=qLayers)
        if self.use_cuda:
            self.Q = self.Q.cuda()
        self.prev = {'observation':None, 'reward':None, 'action':None}
        self.lossTrace = np.array([])

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
        self.epsilon *= self.eld
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
        observations = self.tensorType(observations)
        actions = self.tensorType(actions)
        rewards = self.tensorType(rewards)
        if observations.shape[0] is not rewards.shape[0]:
            raise ValueError(
                'shape[0]s do not match; rows of observations and rewards must align')
        qx = self.oa2qin(observations, actions)
        qs = self.Q(qx).data
        qs = flat(qs)
        # add a zero at the end (will throw away this row after doing tensor math)
        # qcur = torch.cat((qprev[1:], self.tensorType([0])))
        # qcur = qcur.data.view(qcur.size()[0])
        #qy = qcur.data.view(qcur.size()[0]) + self.qlr * (rewards - qprev.data.view(qprev.size()[0])) # SARSA
        #pdb.set_trace()
        qopt = self.tensorType(np.max(np.array([[self.Q(self.oa2qin(o,a)).data[0] for a in range(self.action_space.n)] for o in observations]), axis=1))
        #pdb.set_trace()
        qy = (1-self.qlr) * qs + self.qlr * (rewards + self.gamma(1) * qopt)  # QLearning
        #pdb.set_trace()
        qy = Variable(qy, requires_grad=False)  # remove that row at the end
        # train Q
        loss = _train(self.Q, qx, qy, nIter, self.use_cuda)
        self.lossTrace = np.append(self.lossTrace, loss)
        #self.epsilon = self.eld * observations.shape[0] * self.epsilon


def _train(model, x, y, nIter, use_cuda=True):
    # x = Variable(torch.FloatTensor(x))
    # y = Variable(torch.FloatTensor(y), requires_grad=False)
    # if torch.cuda.is_available() and use_cuda:
    #     model = model.cuda()
    #     x = x.type(torch.cuda.FloatTensor)
    #     y = y.type(torch.cuda.FloatTensor)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
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
            self.mods.append(torch.nn.PReLU())
            self.mods.append(torch.nn.Tanh())
        self.mods.append(torch.nn.Linear(h, o))

    def forward(self, x):
        # return self.fc(x) # it was just x there
        for module in self.mods:
            x = module(x)
        return x
