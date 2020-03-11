import math

from torch import nn


def conv_bn_relu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class GlassesClassifier(nn.Module):
    def __init__(self):
        super(GlassesClassifier, self).__init__()

        self.conv1 = conv_bn_relu(3, 16, kernel=3, padding=1)
        self.conv2 = conv_bn_relu(16, 32, kernel=3, padding=1)
        self.conv3 = conv_bn_relu(32, 32, kernel=3, padding=1)
        self.conv4 = conv_bn_relu(32, 16, kernel=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
