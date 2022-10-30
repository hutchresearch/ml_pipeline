from torch import nn


class DNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.layer1(x)
