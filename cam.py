import torch
import torchvision
from torchvision import transforms as transforms

import numpy as np

import cv2
import matplotlib.pyplot as plt

import errno
import os


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# inspect CAM
def inspect_cam(batch_num, epoch, batch_img, batch_feature, batch_weight):

    # len(batch) = 100
    for i in range(100):

        weight = torch.unsqueeze(batch_weight[i], dim=1)
        weight = torch.unsqueeze(weight, dim=1)
        weight = weight.expand(-1, 5, 5) # [16, 5, 5]
        cam = torch.mul(batch_feature[i], weight) # [16, 5, 5]*[16, 1, 1]->[16, 5, 5]

        # Sumrize the feature maps
        cam_plus = torch.zeros(5, 5)
        for piece in cam:
            cam_plus = cam_plus + piece

        # Normalize ~(0, 1)
        cam_plus = (cam_plus - cam_plus.min())/(cam_plus.max()-cam_plus.min())
        cam_plus = cam_plus.numpy()

        # Rsize as image size (32, 32)
        cam_resize = 255 * cv2.resize(cam_plus, (32, 32))
        cam_resize = cam_resize.astype(np.uint8)
        # print('cam_resize size: {}'.format(cam_resize.shape))

        # Get Heatmap
        heatmap = cv2.applyColorMap(cam_resize, cv2.COLORMAP_JET)

        # Zero out
        heatmap[np.where(cam_resize<=100)] = 0
        heatmap = heatmap.astype(np.uint8)

        img = np.transpose(batch_img[i], (2, 1, 0))
        img = img.numpy()
        # print('img: {}, heatmap: {}'.format(img, heatmap))
        # print('img size: {}, heatmap size: {}'.format(img.shape, heatmap.shape))

        cam_out = cv2.addWeighted(src1=img, alpha=0.8, src2=heatmap, beta=0.2, gamma=0, dtype=cv2.CV_32F)

        output_dir = './out/'
        try:
            os.makedirs(output_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(output_dir):
                pass
            else:
                raise
        plt.imshow(cam_out)
        plt.savefig(output_dir + '{}_{}_epoch{}.png'.format(batch_num, i, epoch))
        print('Save cam in {}'.format(output_dir + '{}_{}_epoch{}.png'.format(batch_num, i, epoch)))

    return