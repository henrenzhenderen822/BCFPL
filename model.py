'''此程序为二分类网络结构,适用于50×50大小的输入'''

import torch
from torch import nn


class Binarynet(nn.Module):

    def __init__(self):
        super(Binarynet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv = nn.Sequential(self.conv1, self.conv2)

        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 60),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(60, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.contiguous().view(x.shape[0], -1)   # 打平操作
        x = self.fc(x)
        return x


# 测试
def main():
    x = torch.randn(2, 3, 50, 50)
    model = Binarynet()
    print(model)
    pred = model(x)
    print(pred.shape)

    print("模型的参数量为: {}  ".format(sum(x.numel() for x in model.parameters())))


if __name__ == '__main__':
    main()