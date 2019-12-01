import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.bottle = nn.Linear(5, 2)
        self.fc2 = nn.Linear(2, 5)
        self.out = nn.Linear(5, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bottle(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class ClassificationNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
