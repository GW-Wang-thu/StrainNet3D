import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    expansion = 1

    def __init__(self):
        super(FusionNet, self).__init__()

        self.Fuser = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=24, kernel_size=1, stride=1, padding=0),  # , padding=4
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=24, out_channels=27, kernel_size=1, stride=1, padding=0),  # , padding=4
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=27, out_channels=27, kernel_size=1, stride=1, padding=0),  # , padding=2
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=27, out_channels=27, kernel_size=1, stride=1, padding=0),  # , padding=2
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=27, out_channels=27, kernel_size=1, stride=1, padding=0),  # , padding=2
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=27, out_channels=27, kernel_size=1, stride=1, padding=0),  # , padding=2
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=27, out_channels=9, kernel_size=1, stride=1, padding=0),  # , padding=2
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, stride=1, padding=0),  # , padding=2
        )

    def forward(self, x):
        '''
        :param x: shape of [batch, 3, h, w], where the 3 channels are 2 optical flow of LD to LR, and 1 disparity of RD to LD
        :return: the same shape as input x, where the 3 channels are UVW
        '''
        return self.Fuser(x)
