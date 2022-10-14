import torch.nn as nn

"""
Implementation for VGG Architectures

Original paper:
    https://arxiv.org/pdf/1409.1556.pdf

Architecture Notes:
    1. First architecture to use guided principles for choosing convolution parameters
    2. All kernel sizes is 3 with padding 1 and stride 1

"""

class VGG(nn.Module):
  def create_features_extractor(self, architecture):
    layers = []
    in_channels = self.in_channels

    for layer in architecture:
      if type(layer) == str:
        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
      else:
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=layer, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(layer),
            nn.ReLU()]
        in_channels = layer

    return nn.Sequential(*layers)
  def __init__(self, in_channels=3, classes=10, architecture=None):
    super(VGG, self).__init__()

    self.in_channels = in_channels
    vgg16_architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,'M', 512, 512, 512,'M', 512, 512, 512,'M']
    self.architecture = architecture if architecture else vgg16_architecture

    self.features_extractor = self.create_features_extractor(architecture)
    self.classifier =  nn.Sequential(
        nn.Flatten(),
        nn.Linear(7 * 7 * 512, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, classes)
    )

  def forward(self, x):
    features = self.features_extractor(x)
    logits = self.classifier(features)
    return logits