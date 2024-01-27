import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, action_space_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 16, 3, 1, 0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(576, 32),
            nn.LeakyReLU(),
            nn.Linear(32, action_space_size),
        )
    def forward(self, x):
        # print(x.shape)
        x = self.layers(x)
        return x
