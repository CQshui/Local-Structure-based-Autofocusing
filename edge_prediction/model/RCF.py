import os, sys
import time

import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _single, _pair, _triple
import torch.nn.functional as F


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel=256, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        # print('avg',avgout.size())  # [16, 256, 1, 1]
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        # print('max',maxout.size())  # [16, 256, 1, 1]
        return self.sigmoid(avgout + maxout)


# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        # print('savg',avgout.size())  # [16, 1, 384, 384]
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # print('smax',maxout.size())  # [16, 1, 384, 384]
        out = torch.cat([avgout, maxout], dim=1)
        # print('s+s',out.size())  # [16, 2, 384, 384]
        out = self.sigmoid(self.conv2d(out))
        # print('ssigmoid',out.size())  # [16, 1, 384, 384]
        return out


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_channel=256, out_channels=64):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel=in_channel)
        self.spatial_attention = SpatialAttentionModule()
        self.out_channel = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        # print('x', x.size())
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        out = residual + out
        # print(out.size())
        out = self.relu(out)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class DilateConv(nn.Module):
    """
    d_rate: dilation rate
    H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)
    set kernel size to 3, stride to 1, padding==d_rate ==> spatial size kept
    """

    def __init__(self, d_rate, in_ch, out_ch):
        super(DilateConv, self).__init__()
        self.d_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                stride=1, padding=d_rate, dilation=d_rate)

    def forward(self, x):
        return self.d_conv(x)


class RCF(nn.Module):
    def __init__(self, device='cuda'):
        super(RCF, self).__init__()
        # lr 1 2 decay 1 0
        self.device = device
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.cbam_1 = CBAM(64, 64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        # self.cbam_2 = CBAM(128, 128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        # self.cbam_3 = CBAM(256, 256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.cbam_4 = CBAM(512, 512)

        # lr 100 200 decay 1 0
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        # self.conv5_1 = DilateConv(d_rate=2, in_ch=512, out_ch=512) # error ! name conv5_1.dconv.weight erro in load vgg16
        # self.conv5_2 = DilateConv(d_rate=2, in_ch=512, out_ch=512)
        # self.conv5_3 = DilateConv(d_rate=2, in_ch=512, out_ch=512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        # self.cbam_5 = CBAM(512, 512)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        # lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(128, 21, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(512, 21, 1, padding=0)

        self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(512, 21, 1, padding=0)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)


    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        # conv1_2 = self.cbam_1(conv1_2)
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        # conv2_2 = self.cbam_2(conv2_2)
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        # conv3_3 = self.cbam_3(conv3_3)
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        # conv4_3 = self.cbam_4(conv4_3)
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))
        # conv5_3 = self.cbam_5(conv5_3)

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)
        # transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).to(self.device)
        weight_deconv3 = make_bilinear_weights(8, 1).to(self.device)
        weight_deconv4 = make_bilinear_weights(16, 1).to(self.device)
        weight_deconv5 = make_bilinear_weights(32, 1).to(self.device)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)

        # center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)

        ### crop way suggested by liu
        # so1 = crop_caffe(0, so1, img_H, img_W)
        # so2 = crop_caffe(1, upsample2, img_H, img_W)
        # so3 = crop_caffe(2, upsample3, img_H, img_W)
        # so4 = crop_caffe(4, upsample4, img_H, img_W)
        # so5 = crop_caffe(8, upsample5, img_H, img_W)
        ## upsample way
        # so1 = F.upsample_bilinear(so1, size=(img_H,img_W))
        # so2 = F.upsample_bilinear(so2, size=(img_H,img_W))
        # so3 = F.upsample_bilinear(so3, size=(img_H,img_W))
        # so4 = F.upsample_bilinear(so4, size=(img_H,img_W))
        # so5 = F.upsample_bilinear(so5, size=(img_H,img_W))

        # 创建边框遮罩
        def create_border_mask(h, w, border_size=5, device='cuda'):
            mask = torch.ones((1, 1, h, w), device=device)
            mask[:, :, :border_size, :] = 0
            mask[:, :, -border_size:, :] = 0
            mask[:, :, :, :border_size] = 0
            mask[:, :, :, -border_size:] = 0
            return mask

        # border_mask = create_border_mask(img_H, img_W, border_size=50, device=x.device)
        border_mask = create_border_mask(img_H, img_W, border_size=50, device=x.device)
        # border_mask = create_border_mask(img_H, img_W, border_size=70, device=x.device)

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)
        # 对每个输出应用遮罩
        # so1 = so1 * border_mask
        # so2 = so2 * border_mask
        # so3 = so3 * border_mask
        # so4 = so4 * border_mask
        # so5 = so5 * border_mask
        # fuse = fuse * border_mask
        # results = [so1, so2, so3, so4, so5, fuse]
        # results = [so1, so2, so3, so4, so5, so5]   # 只取so5作为训练结果
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) * border_mask for r in results]  # 先激活再遮罩
        # results = [torch.sigmoid(r) for r in results]  # 先激活再遮罩

        # threshold = 0.5
        # binary_results = [(torch.sigmoid(r) * border_mask > threshold).float() for r in results]

        return results


class RCF_cut(nn.Module):
    def __init__(self, device='cuda'):
        super(RCF_cut, self).__init__()
        # lr 1 2 decay 1 0
        self.device = device
        # self.se = nn.
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.cbam_1 = CBAM(64, 64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        # self.cbam_2 = CBAM(128, 128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        # self.cbam_3 = CBAM(256, 256)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        # lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(128, 21, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(256, 21, 1, padding=0)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)

        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(3, 1, 1)


    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        # conv1_2 = self.cbam_1(conv1_2)
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        # conv2_2 = self.cbam_2(conv2_2)
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        # conv3_3 = self.cbam_3(conv3_3)
        pool3 = self.maxpool(conv3_3)

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)

        # transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).to(self.device)
        weight_deconv3 = make_bilinear_weights(8, 1).to(self.device)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)

        # center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)

        ### crop way suggested by liu
        # so1 = crop_caffe(0, so1, img_H, img_W)
        # so2 = crop_caffe(1, upsample2, img_H, img_W)
        # so3 = crop_caffe(2, upsample3, img_H, img_W)
        # so4 = crop_caffe(4, upsample4, img_H, img_W)
        # so5 = crop_caffe(8, upsample5, img_H, img_W)
        ## upsample way
        # so1 = F.upsample_bilinear(so1, size=(img_H,img_W))
        # so2 = F.upsample_bilinear(so2, size=(img_H,img_W))
        # so3 = F.upsample_bilinear(so3, size=(img_H,img_W))
        # so4 = F.upsample_bilinear(so4, size=(img_H,img_W))
        # so5 = F.upsample_bilinear(so5, size=(img_H,img_W))

        # 创建边框遮罩
        def create_border_mask(h, w, border_size=5, device='cuda'):
            mask = torch.ones((1, 1, h, w), device=device)
            mask[:, :, :border_size, :] = 0
            mask[:, :, -border_size:, :] = 0
            mask[:, :, :, :border_size] = 0
            mask[:, :, :, -border_size:] = 0
            return mask

        border_mask = create_border_mask(img_H, img_W, border_size=10, device=x.device)
        # border_mask = create_border_mask(img_H, img_W, border_size=70, device=x.device)

        fusecat = torch.cat((so1, so2, so3), dim=1)
        fuse = self.score_final(fusecat)
        # 对每个输出应用遮罩
        # so1 = so1 * border_mask
        # so2 = so2 * border_mask
        # so3 = so3 * border_mask
        # so4 = so4 * border_mask
        # so5 = so5 * border_mask
        # fuse = fuse * border_mask
        # results = [so1, so2, so3, so4, so5, fuse]
        # results = [so1, so2, so3, so4, so5, so5]   # 只取so5作为训练结果
        results = [so1, so2, so3, fuse]
        results = [torch.sigmoid(r) * border_mask for r in results]  # 先激活再遮罩
        # results = [torch.sigmoid(r) for r in results]  # 先激活再遮罩

        return results


def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]


def crop_caffe(location, variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(location)
    y1 = int(location)
    return variable[:, :, y1: y1 + th, x1: x1 + tw]


# make a bilinear interpolation kernel
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(in_channels, out_channels, h, w):
    weights = np.zeros([in_channels, out_channels, h, w])
    if in_channels != out_channels:
        raise ValueError("Input Output channel!")
    if h != w:
        raise ValueError("filters need to be square!")
    filt = upsample_filt(h)
    weights[range(in_channels), range(out_channels), :, :] = filt
    return np.float32(weights)


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def upsample(input, stride, num_channels=1, device='cuda'):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).to(device)
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)


if __name__ == "__main__":
    device = "cuda"
    model = RCF(device=device).to(device)
    input = torch.rand(1, 3, 224, 256).to(device)
    output0 = model(input)

    st = time.time()
    output = model(input)
    ed = time.time()

    print(ed - st)
