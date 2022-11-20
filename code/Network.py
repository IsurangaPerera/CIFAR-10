import torch.nn as nn
import torch.nn.functional as F


class ResUnitBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super(ResUnitBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_features != out_features:
            self.projection = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.projection = None

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.projection(out) if self.projection is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNetV2Prop(nn.Module):
    def __init__(self, block):
        super(ResNetV2Prop, self).__init__()
        self.start_features = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convStack1 = self._stack_layer(block, 64, 2, stride=1)
        self.convStack2 = self._stack_layer(block, 128, 2, stride=2)
        self.convStack3 = self._stack_layer(block, 256, 2, stride=2)
        self.convStack4 = self._stack_layer(block, 512, 2, stride=2)
        self.linear = nn.Linear(512, 10)

    def _stack_layer(self, block, out_features, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i != 0:
                stride = 1

            if i == num_blocks - 1:
                self.start_features = out_features
            layers.append(block(self.start_features, out_features, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.convStack1(out)
        out = self.convStack2(out)
        out = self.convStack3(out)
        out = self.convStack4(out)
        out = F.adaptive_avg_pool2d(out, output_size=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
