"""
2019-6-6 by kylin
Email:dragonsaint@qq.com
"""

import torch
import torch.nn.functional as F

#replace following class code with an easy sequential network
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)   #hidden layer
        self.predict = torch.nn.Linear(n_hidden,n_output)   #output layer

    def forward(self,x):
        x = F.relu(self.hidden(x))  #activation function for hidden layer
        x = self.predict(x)         #linear output
        return x


net1 = Net(1,10,1)


#easy and fast way to build your network
net2 = torch.nn.Sequential(torch.nn.Linear(1,10),torch.nn.ReLU(),torch.nn.Linear(10,1))

print(net1) #net1 architecture

print(net2) #net2 architecture

"""
print result:
Net(
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=1, bias=True)
)
Sequential(
  (0): Linear(in_features=1, out_features=10, bias=True)
  (1): ReLU()
  (2): Linear(in_features=10, out_features=1, bias=True)
)
"""
