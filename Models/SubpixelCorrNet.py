import torch.nn as nn
import torch
import torch.nn.functional as F


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        # nn.LeakyReLU(0.1,inplace=True)
        nn.Tanh()
    )


def deconv1(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
    )


class SubpixelCorrNet(nn.Module):
    def __init__(self, batchNorm=True):
        super(SubpixelCorrNet, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=7, stride=1, padding=3),      # , padding=4
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=2, padding=2),     # , padding=3
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),    # , padding=2
            nn.LeakyReLU(0.2),
        )

        self.correlator1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),

            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.Tanh()
        )

        self.dslayer = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)

        self.correlator3 = nn.Sequential(

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),

            deconv(256, 64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            deconv1(32, 8),
            nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        ref_features = self.extractor(x[:, 0, :, :].unsqueeze(1))
        def_features = self.extractor(x[:, 1, :, :].unsqueeze(1))
        input_tensor = torch.cat((ref_features, def_features), dim=1)
        disps1 = self.correlator1(input_tensor)
        ds1 = self.dslayer(input_tensor)
        ds2 = self.downsample(ds1)
        disps2 = self.correlator3(ds2)
        cat_disp = torch.cat((disps1, disps2), dim=1)
        out = self.refine(cat_disp)
        if self.training:
            return out, disps1, disps2
        else:
            return out


def loss_flow(output_list, label, weights):
    b, _, h, w = label.size()
    loss = []
    meanloss = 0
    for i in range(len(output_list)):
      upsampled_output = F.interpolate(output_list[i], (h, w), mode='bilinear', align_corners=False)
      temp_loss = torch.norm(label - upsampled_output, dim=1, p=2).mean() * weights[i]
      loss.append(temp_loss)
      meanloss += temp_loss
    loss.append(meanloss)
    return loss


def loss_disp(output_list, label, weights):
    b, _, h, w = label.size()
    loss = []
    meanloss = 0
    for i in range(len(output_list)):
      upsampled_output = F.interpolate(output_list[i], (h, w), mode='bilinear', align_corners=False)
      temp_loss = torch.norm(label[0, :, :, :] - upsampled_output[0, 0, :, :], dim=1, p=2).mean() * weights[i]
      loss.append(temp_loss)
      meanloss += temp_loss
    loss.append(meanloss)
    return loss