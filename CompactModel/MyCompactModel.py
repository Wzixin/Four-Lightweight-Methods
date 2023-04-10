import torch
import torch.nn as nn
import torch.nn.functional as F


class MyCompactNet(nn.Module):
    def __init__(self, num_classes=10, pruning=False):
        super(MyCompactNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1, groups=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, 3, 2, 1, groups=16)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, 2, 1, groups=16)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16*4*4, num_classes)

    def forward(self, x):
        out = self.conv1(x)  # (卷积+BN)*3
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)    # 非线性激活层
        out = out.view(out.size(0), -1)
        out = self.fc(out)   # 全连接层
        return out


