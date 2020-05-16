#   P,R,O网络
#   三个独立的网络
#   可以同时训练

import torch.nn as nn
import torch.nn.functional as F


#   P网络
class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU()
        )
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        x = self.pre_layer(x)
        #   置信度用sigmoid激活（用BCELoss时要先使用sigmoid激活）
        cond = F.sigmoid(self.conv4_1(x))
        #   偏移量不需要激活，原样输出
        offset = self.conv4_2(x)
        return cond, offset


#   R网络
class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(28, 48, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 64, 2, 1),
            nn.PReLU()
        )
        self.conv4 = nn.Linear(64 * 3 * 3, 128)
        self.prelu4 = nn.PReLU()
        self.conv5_1 = nn.Linear(128, 1)
        self.conv5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        #   N,V
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        #   置信度（sigmoid激活）
        label = F.sigmoid(self.conv5_1(x))
        #   偏移量
        offset = self.conv5_2(x)
        return label, offset


#   O网络
class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 64, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.PReLU()
        )
        self.conv5 = nn.Linear(128 * 3 * 3, 256)
        self.prelu5 = nn.PReLU()
        self.conv6_1 = nn.Linear(256, 1)
        self.conv6_2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        #   置信度（sigmoid激活）
        label = F.sigmoid(self.conv6_1(x))
        #   偏移量
        offset = self.conv6_2(x)
        return label, offset
