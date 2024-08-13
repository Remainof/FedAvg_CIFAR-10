import torch.nn as nn
import torch.nn.functional as F

# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一个卷积层：输入通道数3，输出通道数6，卷积核大小5x5
        self.conv1 = nn.Conv2d(3, 6, 5)

        # 定义最大池化层：窗口大小2x2
        self.pool = nn.MaxPool2d(2, 2)

        # 定义第二个卷积层：输入通道数6，输出通道数16，卷积核大小5x5
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 定义全连接层：输入节点数16*5*5，输出节点数120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # 定义全连接层：输入节点数120，输出节点数84
        self.fc2 = nn.Linear(120, 84)

        # 定义全连接层：输入节点数84，输出节点数10（对应10个类别）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 定义前向传播过程
        x = self.pool(F.relu(self.conv1(x)))  # 卷积 -> ReLU激活 -> 池化
        x = self.pool(F.relu(self.conv2(x)))  # 卷积 -> ReLU激活 -> 池化
        x = x.view(-1, 16 * 5 * 5)  # 展平操作
        x = F.relu(self.fc1(x))  # 全连接层 -> ReLU激活
        x = F.relu(self.fc2(x))  # 全连接层 -> ReLU激活
        x = self.fc3(x)  # 输出层，不使用激活函数，因为CrossEntropyLoss包含了softmax
        return x
