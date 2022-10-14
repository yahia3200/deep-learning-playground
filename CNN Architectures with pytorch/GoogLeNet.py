"""
Implementation for GoogLeNet Architectures

Original paper:
    https://arxiv.org/pdf/1409.4842.pdf

Architecture Notes:
    1. Introduced using a steam network at the beginning to reduce width and 
    height for inputs which lead to more efficient network
    2. Introduced 1*1 kernel, Inception module and global average pooling
    3. To train the network they used an auxiliary classifiers at several intermediate
    points so that the loss can propagate well, no need for this when using batch normalization

"""

import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class inception(nn.Module):
  def __init__(self, in_channels, out_1, reduction_3, out_3, reduction_5, out_5, out_pool):
        super(inception, self).__init__()

        self.branch_1 = conv_block(in_channels=in_channels, out_channels=out_1, kernel_size=1)
        self.branch_2 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=reduction_3, kernel_size=1),
            conv_block(in_channels=reduction_3, out_channels=out_3, kernel_size=3, padding=1)
        )
        self.branch_3 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=reduction_5, kernel_size=1),
            conv_block(in_channels=reduction_5, out_channels=out_5, kernel_size=5, padding=2)
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels=in_channels, out_channels=out_pool, kernel_size=1)
        )

  def forward(self, x):
    return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], 1)

class GoogLeNet(nn.Module):
  def __init__(self, in_channels=3, classes=10):
    super(GoogLeNet, self).__init__()

    self.steam_network = nn.Sequential(
        conv_block(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3, stride=2),
        conv_block(in_channels=64, out_channels=192, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # In this order: in_channels, out_1, reduction_3, out_3, reduction_5, out_5, out_pool
    self.inception3a = inception(192, 64, 96, 128, 16, 32, 32)
    self.inception3b = inception(256, 128, 128, 192, 32, 96, 64)

    self.inception4a = inception(480, 192, 96, 208, 16, 48, 64)
    self.inception4b = inception(512, 160, 112, 224, 24, 64, 64)
    self.inception4c = inception(512, 128, 128, 256, 24, 64, 64)
    self.inception4d = inception(512, 112, 144, 288, 32, 64, 64)
    self.inception4e = inception(528, 256, 160, 320, 32, 128, 128)

    self.inception5a = inception(832, 256, 160, 320, 32, 128, 128)
    self.inception5b = inception(832, 384, 192, 384, 48, 128, 128)

    self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

    self.classifier = nn.Sequential(
        nn.AvgPool2d(kernel_size=7, stride=1),
        nn.Flatten(),
        nn.Dropout(p=0.4),
        nn.Linear(1024, classes)
    )

  def forward(self, x):
    reduced_input = self.steam_network(x)
    
    features = self.inception3a(reduced_input)
    features = self.inception3b(features)
    features = self.maxpool(features)

    features = self.inception4a(features)
    features = self.inception4b(features)
    features = self.inception4c(features)
    features = self.inception4d(features)
    features = self.inception4e(features)
    features = self.maxpool(features)

    features = self.inception5a(features)
    features = self.inception5b(features)

    logits = self.classifier(features)
    return logits
