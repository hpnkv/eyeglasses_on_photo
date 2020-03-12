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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
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


class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResBlock, self).__init__()

        self.conv_bn_relu_1 = conv_bn_relu(n_channels, n_channels, 3, 1)
        self.conv_2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(x + self.bn2(self.conv_2(self.conv_bn_relu_1(x))))


class ResidualGlassesClassifier(nn.Module):
    def __init__(self):
        super(ResidualGlassesClassifier, self).__init__()

        self.conv1 = conv_bn_relu(3, 16, kernel=3, padding=1)
        self.conv2 = ResBlock(16)
        self.conv3 = conv_bn_relu(16, 32, kernel=3, padding=1)
        self.conv4 = ResBlock(32)
        self.conv5 = conv_bn_relu(32, 16, kernel=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv4(self.conv3(x))
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
