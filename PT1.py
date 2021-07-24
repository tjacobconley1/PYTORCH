import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import  Dataset, DataLoader
from torchvision import transforms, datasets

#define datasets (will be serial read I believe for USB camera)
train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

class Net(nn.Module):
        def _init_(self):
                super()._init_()
                self.fc1 = nn.Linear(28*28, 64)
                #because out put of ^^^^ is 64
                self.fc2 = nn.Linear(64, 64)
                self.fc1 = nn.Linear(64, 64)
                #because batch_size = 10
                self.fc1 = nn.Linear(64, 10)

net = Net()
print(net)
