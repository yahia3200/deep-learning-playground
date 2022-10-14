"""
    Building a simple CNN
    Unlike tensorflow we need to keep track of dimensions 
    For more Info about convolution in pytorch
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
"""

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels, classes):
        super(CNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels=8,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), nn.Flatten(),
            nn.Linear(16 * 7 * 7, classes))

    def forward(self, x):
        logits = self.layers(x)
        return logits
