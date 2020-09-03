from ops import *
import torch.nn as nn
import torch

""" Layers """


class Interpolate(nn.Module):
    def __init__(self, scalefactor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scalefactor = scalefactor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scalefactor,
                        mode=self.mode, align_corners=False)
        return x


class DownSampBlock(nn.Module):
    def __init__(self, i, o, k, s, p):
        super(DownSampBlock, self).__init__()
        self.conv = conv_layer(i, o, k=k, s=s, p=p)
        self.batchNorm = batchNorm(o)
        self.ReLU = ReLU()

    def forward(self, x):
        y = self.ReLU(self.batchNorm(self.conv(x)))
        return y


class ResBlock(nn.Module):
    def __init__(self, i, o, k, s, p):
        super(ResBlock, self).__init__()
        self.conv1 = conv_layer(i, o, k=k, s=s, p=p)
        self.batchNorm1 = batchNorm(o)
        self.conv2 = conv_layer(i, o, k=k, s=s, p=p)
        self.batchNorm2 = batchNorm(o)
        self.ReLU = ReLU()

    def forward(self, x):
        y = self.ReLU(self.batchNorm1(self.conv1(x)))
        y = self.batchNorm2(self.conv2(x))
        y += x
        return y


class UpSampBlock(nn.Module):
    def __init__(self, i, o, k, s, p):
        super(UnSampBlock, self).__init__()
        self.upSampling = upSampling()
        self.conv = conv_layer(i, o, k=k, s=s, p=p)
        self.batchNorm = batchNorm(o)
        self.ReLU = ReLU()

    def forward(self, x):
        y = self.ReLU(self.batchNorm(self.conv(self.upSampling(x))))
        return y


def conv_layer(i, o, k=3, s=1, p=1):
    return nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=p, bias=False)


"""Activation Functions"""


def ReLU():
    return nn.ReLU()


def tanH():
    return nn.Tanh()


""" Batch Normalization """


def batchNorm(channels):
    return nn.BatchNorm2d(channels)


"""Loss Functions"""


def MSE(a, b):
    criterion = nn.MSELoss()
    return criterion(a, b)


def MAE(a, b):
    return abs(a-b).sum()/a.numel()


def minmaxLoss_onepoint(img):
    max1, _ = img.max(2)
    max2, _ = max1.max(2)

    min1, _ = img.min(2)
    min2, _ = min1.min(2)

    loss = max2-min2
    sum = loss.sum(0)

    return sum


def minmaxLoss_rowcolumn(img):

    for i in range(img.size()[0]):
        plane = img[i, :, :, :]
        meanval = plane.mean()
        plane = plane - meanval
        img[i, :, :, :] = abs(plane)

    sum = img.sum()

    return sum


def upSampling():
    return nn.UpsamplingBilinear2d(scale_factor=2)


def n1(tensor):
    if torch.sub(torch.max(tensor), torch.min(tensor)) == 0:
        tensor = tensor / tensor
    else:
        tensor = torch.div(
            torch.sub(
                tensor,
                torch.min(tensor)
            ),
            torch.sub(
                torch.max(tensor),
                torch.min(tensor)
            )
        )
    return tensor


def n2(tensor):
    if torch.sub(torch.max(tensor), torch.min(tensor)) == 0:
        tensor = tensor / tensor
    else:
        tensor = torch.div(tensor, torch.max(tensor))
    return tensor
