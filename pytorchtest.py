import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
data = pd.read_csv('forestfires.csv').as_matrix()
dataD = np.delete(np.delete(data,2,axis=1), 2, axis=1).astype(np.dtype(float))
xn = dataD[:,:-2]
m,s = np.mean(xn,axis=0), np.std(xn,axis=0)
xn -= m
xn /= s
yn = dataD[:,-2]
m,s = np.mean(yn,axis=0), np.std(yn,axis=0)
yn -= m
yn /= s
x = Variable(torch.from_numpy(xn).type(torch.FloatTensor))
y = Variable(torch.from_numpy(yn).type(torch.FloatTensor),requires_grad=False)
class Model(torch.nn.Module):
    
    def __init__(self,n,h,o):
        super(Model, self).__init__()
        self.mods = torch.nn.ModuleList([])
        self.mods.append(torch.nn.Linear(n, h))
        self.mods.append(torch.nn.ReLU())
        self.mods.append(torch.nn.Tanh())
        self.mods.append(torch.nn.ReLU())
        self.mods.append(torch.nn.Tanh())
        self.mods.append(torch.nn.ReLU())
        self.mods.append(torch.nn.Linear(h,o))
        
    def forward(self, x):
        #return self.fc(x) # it was just x there
        for module in self.mods:
            x = module(x)
        return x
# Code in file nn/two_layer_net_module.py
import torch
import time
tstart = time.time()
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
H, D_out = 500, 1
use_cuda = True

# Construct our model by instantiating the class defined above
model = Model(x.size()[1],H,D_out)
if torch.cuda.is_available() and use_cuda:
    model = model.cuda()
    x = x.type(torch.cuda.FloatTensor)
    y = y.type(torch.cuda.FloatTensor)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
lossTrace = []
for t in range(6000):
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
    
print('Trained in {} seconds'.format(time.time() - tstart))
plt.plot(lossTrace)
plt.plot(y.data.cpu().numpy())
plt.plot(model(x).data.cpu().numpy())