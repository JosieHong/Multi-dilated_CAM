import torch
import torch.nn as nn
import torch.nn.functional as func

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.gap = nn.AvgPool2d(kernel_size = 5)
        # CPU
        self.gap_w = torch.rand((16, 10), requires_grad=True)
        # GPU
        # self.gap_w = torch.rand((16, 10), requires_grad=True).cuda()

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        self.feature = x # [100, 16, 5, 5]
        gap = torch.squeeze(self.gap(x)) # [100, 16, 1, 1]->[100, 16]
        self.weight = gap
        x = torch.matmul(gap, self.gap_w) # [100, 16]*[16, 10]->[100, 10]

        return x