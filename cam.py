import torch
import torchvision
from torchvision import transforms as transforms

import numpy as np

import cv2

# inspect CAM
def inspect_cam(batch_img, batch_feature, batch_weight):

    featur_size = batch_feature.size()[2]
    image_size = batch_img.size()[2]

    weight = torch.unsqueeze(batch_weight, dim=1) # [16, 1]
    weight = torch.unsqueeze(weight, dim=1) # [16, 1, 1]
    weight = weight.expand(-1, featur_size, featur_size) # [16, featur_size, featur_size]
    # [16, featur_size, featur_size]*[16, 1, 1]->[16, featur_size, featur_size]
    cam = torch.mul(batch_feature, weight)

    # Sumrize the feature maps
    cam_plus = torch.zeros(featur_size, featur_size)
    for piece in cam:
        cam_plus = cam_plus + piece

    # Normalize ~(0, 1)
    cam_plus = (cam_plus - cam_plus.min())/(cam_plus.max()-cam_plus.min())
    cam_plus = cam_plus.numpy()

    # Rsize as image size (32, 32)
    cam_resize = 255 * cv2.resize(cam_plus, (image_size, image_size))
    cam_resize = cam_resize.astype(np.uint8)
    # print('cam_resize size: {}'.format(cam_resize.shape))

    # Get Heatmap
    heatmap = cv2.applyColorMap(cam_resize, cv2.COLORMAP_JET)

    # Zero out
    heatmap[np.where(cam_resize<=150)] = 0
    heatmap = heatmap.astype(np.uint8)

    img = np.transpose(batch_img, (1, 2, 0))
    img = img.numpy()
    # print('img: {}, heatmap: {}'.format(img, heatmap))
    # print('img size: {}, heatmap size: {}'.format(img.shape, heatmap.shape))

    cam_out = cv2.addWeighted(src1=img, alpha=0.9, src2=heatmap, beta=0.1, gamma=0, dtype=cv2.CV_32F)

    return cam_out