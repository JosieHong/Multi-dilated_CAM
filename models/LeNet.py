import torch
import torch.nn as nn
import torch.nn.functional as func


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = func.relu(self.conv1(x))
#         x = func.max_pool2d(x, 2)
#         x = func.relu(self.conv2(x))
#         x = func.max_pool2d(x, 2)
#         x = x.view(x.size(0), -1)
#         x = func.relu(self.fc1(x))
#         x = func.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.gap = nn.AvgPool2d(kernel_size = 5)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2) # [100, 16, 5, 5]
        self.feature = x
        # Remove the dimension with one value
        gap = torch.squeeze(self.gap(x)) # [100, 16, 1, 1]->[100, 16]
        gap_w = torch.rand((100, 16, 10), requires_grad=True) # [100, 16, 10]
        x = torch.matmul(gap, gap_w)

        print('gap_size = {}, gap_w_size = {}, x_size = {}'.format(gap.size(), gap_w.size(), x.size()))
        return x