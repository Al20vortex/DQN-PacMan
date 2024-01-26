import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, action_space_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 16, 1, 1, 0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_space_size),
        )
    def forward(self, x):
        # print(x.shape)
        x = self.layers(x)
        return x
