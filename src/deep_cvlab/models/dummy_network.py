import torch.nn as nn

SOME_RELEVANT_CONSTANT = 3

class DummyNet(nn.Module):
    def __init__(self, num_layers, nonlin_type='leakyrelu', **kwargs):
        super(DummyNet, self).__init__()
        self.num_layers = num_layers

        if nonlin_type == 'relu':
            nonlin = nn.ReLU() 
        elif nonlin_type == 'leakyrelu':
            # then "leaky_slope" also must be specified
            nonlin = nn.LeakyReLU(kwargs['leaky_slope']) 
        else:
            raise ValueError(f'Check nonlin_type! Options: "relu" | "leakyrelu", yours : {nonlin_type} ')

        net = [
            nn.Linear(self.num_layers, self.num_layers),
            nonlin,
            nn.Linear(self.num_layers, self.num_layers),
        ]
        self.net = nn.Sequential(*net)

        self.num_iterations = SOME_RELEVANT_CONSTANT

    def forward(self, x):
        for i in range(self.num_iterations):
            x = self.net(x)
        return x
