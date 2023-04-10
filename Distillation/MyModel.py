import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(MyConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)

    def forward(self, Activation):
        # 获得初始化权重
        real_weights = self.weights.view(self.shape)
        # 卷积输出
        out = F.conv2d(Activation, real_weights, stride=self.stride, padding=self.padding)
        return out


class MyTeacherNet(nn.Module):
    def __init__(self, num_classes=10, pruning=False):
        super(MyTeacherNet, self).__init__()

        self.conv1 = MyConv(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = MyConv(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = MyConv(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64*4*4, num_classes)

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


class MyStudentNet(nn.Module):
    def __init__(self, num_classes=10, pruning=False):
        super(MyStudentNet, self).__init__()

        self.conv1 = MyConv(1, 2, kernel_size=3, stride=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = MyConv(2, 4, kernel_size=3, stride=3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4*4*4, num_classes)

    def forward(self, x):
        out = self.conv1(x)  # (卷积+BN)*3
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)    # 非线性激活层
        out = out.view(out.size(0), -1)
        out = self.fc(out)   # 全连接层
        return out


