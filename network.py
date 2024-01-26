import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, action_space_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, 3, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, 1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4624, action_space_size),
        )
    def forward(self, x):
        # print(x.shape)
        x = self.layers(x)
        return x
