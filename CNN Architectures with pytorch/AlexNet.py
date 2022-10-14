import torch.nn as nn

"""
Implementation for AlexNet Architectures

Original paper:
    https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

Architecture Notes:
    1. AlexNet was first CNN to win ImageNet challenge in 2012
    2. at that time the network didn't fit in one gpu and the training
    was done using two
    3. AlexNet was first to use ReLu activision functions
    4. The writers also used some normalization called Local Response Normalization
    5. They also used some tricks in training such as:  Data Augmentation, Dropout and weight decay 

"""


class AlexNet(nn.Module):
    def __init__(self, in_channels=3, classes=10):
        super(AlexNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=96,
                kernel_size=(11, 11),
                stride=4,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=(5, 5),
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(
                in_channels=256,
                out_channels=348,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=348,
                out_channels=348,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=348,
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(5 * 5 * 256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
