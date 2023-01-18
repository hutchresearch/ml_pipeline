from torch import nn


class DNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()

        # Define the activation function and the linear functions
        self.act = nn.ReLU()
        self.in_linear = nn.Linear(in_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, out_size)

    def forward(self, x):

        # Send x through first linear layer and activation function
        x = self.act(self.in_linear(x))

        # Return x through the out linear function
        return self.out_linear(x)
