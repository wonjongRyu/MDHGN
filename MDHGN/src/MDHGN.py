from ops import *
import torch
import math


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class MDHGN(nn.Module):
    def __init__(self, args):
        super(MDHGN, self).__init__()

        self.num_of_blocks = args.num_of_blocks  # 15
        self.ch = args.num_of_initChannel  # 128
        self.sz = args.size_of_images  # 512

        self.conv1 = conv_layer(1, self.ch, k=7, s=1, p=3)

        self.DownSampBlocks = self.make_DownSampBlocks(num_blocks=2)
        self.ResBlocks = self.make_ResBlocks(num_blocks=15)
        self.UpSampBlocks = self.make_UpSampBlocks(num_blocks=2)

        self.conv2 = conv_layer(self.ch, 1, k=7, s=1, p=3)
        self.conv3 = conv_layer(self.ch, 1, k=7, s=1, p=3)

        self.ReLU = ReLU()
        self.tanH = tanH()

    def make_DownSampBlocks(self, num_blocks, k=4, s=2, p=1):
        layers = [DownSampBlock(self.ch, self.ch*2, k=k, s=s, p=p)]
        self.ch = self.ch*2
        for _ in range(num_blocks - 1):
            layers.append(DownSampBlock(self.ch, self.ch*2, k=k, s=s, p=p))
            self.ch = self.ch*2

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def make_ResBlocks(self, num_blocks, k=3, s=1, p=1):
        layers = [ResBlock(self.ch, self.ch, k=k, s=s, p=p)]
        for _ in range(num_blocks - 1):
            layers.append(ResBlock(self.ch, self.ch, k=k, s=s, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def make_UpSampBlocks(self, num_blocks, k=3, s=1, p=1):
        layers = [UpSampBlock(self.ch, self.ch//2, k=k, s=s, p=p)]
        self.ch = self.ch//2
        for _ in range(num_blocks - 1):
            layers.append(UpSampBlock(self.ch, self.ch//2, k=k, s=s, p=p))
            self.ch = self.ch//2

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def forward(self, x):
        """ forward function"""

        x = self.conv1(x)
        x = self.DownSampBlocks(x)
        x = self.ResBlocks(x)
        x = self.UpSampBlocks(x)

        real = (self.tanH(self.conv2(x))+1)/2
        imag = (self.tanH(self.conv3(x))+1)/2

        return real, imag
