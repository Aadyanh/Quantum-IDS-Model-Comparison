import torch.nn as nn


class LinearSVM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def l2(self):
        return (self.lin.weight ** 2).sum()

    def forward(self, x):
        return self.lin(x).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def l2(self):
        return 0.0

    def forward(self, x):
        return self.net(x).squeeze(-1)
