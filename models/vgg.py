import torch
import torch.nn as nn


cfg = {
    # 'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # Remove one conv block to enlarge the revolution of the feature maps
    'VGG16_modi': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    # 'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, 10)
        self.gap = nn.AvgPool2d(kernel_size = 4)
        # CPU
        self.gap_w = torch.rand((512, 10), requires_grad=True)
        # GPU
        # gap_w = torch.rand((512, 10), requires_grad=True).cuda()

    def forward(self, x):
        out = self.features(x)
        # out = out.view(out.size(0), -1)
        self.feature = out # [100, 512, 4, 4]
        gap = torch.squeeze(self.gap(out)) # [100, 512, 1, 1] -> [100, 512]
        self.weight = gap
        out = torch.matmul(gap, self.gap_w) # [100, 512]*[512, 10] -> [100, 10]

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        # Remove one pooling layer to enlarge the revolution of the feature maps
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)