import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


class FusionNet_S(nn.Module):
    expansion = 1
    def __init__(self, params="./EXP_related/states.json", cropbox=[500, 2500, 1000, 3000], transpose_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
        super(FusionNet_S, self).__init__()
        self.cropbox = cropbox
        self.Transpose_matrix = transpose_matrix

        with open(params, 'r') as f:
            self.params = json.load(f)
        self.Tx_r = self.params["RTX"]
        self.Tz_r = self.params["RTZ"]
        self.cx_r = self.params["RCX"]
        self.fx_r = self.params["RFX"]

        self.cx_l = self.params["LCX"]
        self.cy_l = self.params["LCY"]
        self.fx_l = self.params["LFX"]
        self.fy_l = self.params["LFY"]
        self.RRot = np.array(self.params["RRot"])
        self.shift_LX, self.shift_LY = self._generate_l_array(self.cropbox)
        self.proj_disparity_y = torch.from_numpy(
            np.zeros(shape=(self.cropbox[1] - self.cropbox[0], self.cropbox[3] - self.cropbox[2]), dtype="float32")).cuda()

        self.resolve_layer = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=27, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv2d(in_channels=27, out_channels=27, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=27, out_channels=3, kernel_size=1, stride=1, padding=0)
        )
        self.input_layer = nn.Conv2d(in_channels=3, out_channels=27, kernel_size=3, stride=1, padding=1)

        self.block1 = nn.Conv2d(in_channels=27, out_channels=27, kernel_size=3, stride=1, padding=1)
        self.block2 = nn.Conv2d(in_channels=27, out_channels=27, kernel_size=3, stride=1, padding=1)
        self.block3 = nn.Conv2d(in_channels=27, out_channels=27, kernel_size=3, stride=1, padding=1)
        self.out_layer = nn.Conv2d(in_channels=27, out_channels=3, kernel_size=3, stride=1, padding=1)

    def _generate_l_array(self, crop):
        lx = torch.from_numpy(np.expand_dims(np.arange(crop[2], crop[3], 1).astype("float32"), 0).repeat(crop[1] - crop[0], axis=0)).cuda()
        ly = torch.from_numpy(np.expand_dims(np.arange(crop[0], crop[1], 1).astype("float32"), 1).repeat(crop[3] - crop[2], axis=1)).cuda()
        return lx, ly

    def forward(self, flow_output, disp0, disp1, key=0):
        ''' key = 0, theoretically calculate u, v and w using disparities and optical flow first
            key = 1, flow_output->raw u, disp0->raw v, disp1->raw w '''
        if key == 0:
            now = time.perf_counter()
            right_projed_coord_x = self.shift_LX + disp0[0, 0, :, :]
            right_projed_coord_y = self.shift_LY + self.proj_disparity_y
            right_coord_x = (self.Transpose_matrix[0, 0] * right_projed_coord_x + self.Transpose_matrix[
                0, 1] * right_projed_coord_y + self.Transpose_matrix[0, 2]) \
                            / (self.Transpose_matrix[2, 0] * right_projed_coord_x + self.Transpose_matrix[
                2, 1] * right_projed_coord_y + self.Transpose_matrix[2, 2])
            zw_0 = (self.Tx_r - self.Tz_r * (right_coord_x + 1.0 - self.cx_r) / self.fx_r) / \
                   (((right_coord_x + 1.0 - self.cx_r) / self.fx_r) *
                    (self.RRot[2, 0] * (self.shift_LX + 1.0 - self.cx_l) / self.fx_l + self.RRot[2, 1] * (
                            self.shift_LY + 1.0 - self.cy_l) / self.fy_l + self.RRot[2, 2])
                    - (self.RRot[0, 0] * (self.shift_LX + 1.0 - self.cx_l) / self.fx_l + self.RRot[0, 1] * (
                                   self.shift_LY + 1.0 - self.cy_l) / self.fy_l + self.RRot[0, 2]))
            xw_0 = zw_0 * (self.shift_LX + 1.0 - self.cx_l) / self.fx_l
            yw_0 = zw_0 * (self.shift_LY + 1.0 - self.cy_l) / self.fy_l

            left_coord_x_1 = self.shift_LX + flow_output[0, 0, :, :]
            left_coord_y_1 = self.shift_LY + flow_output[0, 1, :, :]
            right_projed_coord_x_1 = left_coord_x_1 + disp1[0, 0, :, :]
            right_projed_coord_y_1 = left_coord_y_1 + self.proj_disparity_y
            right_coord_x_1 = (self.Transpose_matrix[0, 0] * right_projed_coord_x_1 + self.Transpose_matrix[
                0, 1] * right_projed_coord_y_1 + self.Transpose_matrix[0, 2]) / (
                                      self.Transpose_matrix[2, 0] * right_projed_coord_x_1 + self.Transpose_matrix[
                                  2, 1] * right_projed_coord_y_1 + self.Transpose_matrix[2, 2])
            zw_1 = (self.Tx_r - self.Tz_r * (right_coord_x_1 + 1.0 - self.cx_r) / self.fx_r) / \
                   (((right_coord_x_1 + 1.0 - self.cx_r) / self.fx_r) * (
                           self.RRot[2, 0] * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l + self.RRot[2, 1] * (
                           left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l + self.RRot[2, 2]) -
                    (self.RRot[0, 0] * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l + self.RRot[0, 1] * (
                            left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l +
                     self.RRot[0, 2]))
            xw_1 = zw_1 * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l
            yw_1 = zw_1 * (left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l

            u = (xw_1 - xw_0)
            v = (yw_1 - yw_0)
            w = (zw_1 - zw_0)

            print("Theoretically: ", time.perf_counter() - now)

        else:
            u = flow_output
            v = disp0
            w = disp1

        now = time.perf_counter()
        tensor_disp = torch.cat([u.unsqueeze(0), v.unsqueeze(0), w.unsqueeze(0)], dim=0).unsqueeze(0)
        input_x = self.input_layer(tensor_disp)
        out_1 = self.block1(input_x)
        out_2 = self.block2(out_1) + out_1
        out_3 = self.block3(out_2) + out_2
        out = self.out_layer(out_3) + tensor_disp
        print("Refine: ", time.perf_counter() - now)
        return tensor_disp, out     # return raw u, v and w; refined u, v and w


def loss_fn(batch, label, weights):
    error = label - batch
    loss = 0.0
    for i in range(3):
        loss += torch.norm(error[0, i, :, :] * weights[i], p=2).mean()
    return loss

