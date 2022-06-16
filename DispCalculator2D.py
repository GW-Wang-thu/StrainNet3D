import time

import numpy as np
import torch
from Models import SubpixelCorrNet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import json

class DispCalculator:

    def __init__(self, cropbox=[0, -1, 0, -1], flow_model_params=''):
        self.cropbox = cropbox
        self.inputTrans = transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))
        self.flow_model = SubpixelCorrNet.SubpixelCorrNet().cuda()
        self.flow_model.load_state_dict(torch.load(flow_model_params))
        self.flow_model.eval()
        self.GaussianFiltersize = 5
        self.h = self.cropbox[1] - self.cropbox[0]
        self.w = self.cropbox[3] - self.cropbox[2]

    def _Get_Avg_Matrix(self, img, windsize=100):
        imsize = img.shape
        avg_matrix = []
        for i in range(imsize[0] // windsize):
            start_x = i * windsize
            t_list = []
            for j in range(imsize[1] // windsize):
                start_y = j * windsize
                t_block = img[start_x:start_x + windsize, start_y:start_y + windsize]
                avg = np.average(t_block)
                t_list.append(avg)
            avg_matrix.append(t_list)
        avg_matrix = cv2.resize(np.array(avg_matrix, dtype="uint8"), dsize=(imsize[1], imsize[0]),
                                interpolation=cv2.INTER_CUBIC)

        return avg_matrix, np.average(avg_matrix)

    def runcase(self, lr, rr):
        lr = lr[self.cropbox[0]:self.cropbox[1], self.cropbox[2]:self.cropbox[3]]
        rr = rr[self.cropbox[0]:self.cropbox[1], self.cropbox[2]:self.cropbox[3]]

        lr = (lr-200) * (lr < 200) + 200
        rr = (rr-200) * (rr < 200) + 200

        self.avg_matrix_l, self.avg_l = self._Get_Avg_Matrix(lr[:, ])
        self.avg_matrix_r, self.avg_r = self._Get_Avg_Matrix(rr[:, ])

        self.lr = 0.5 * (self.avg_r + self.avg_l) * lr / self.avg_matrix_l
        self.lr = (self.lr - 255) * (self.lr <= 255) + 255
        self.lr = cv2.GaussianBlur(self.lr.astype('uint8'), ksize=(self.GaussianFiltersize, self.GaussianFiltersize),
                                   sigmaX=self.GaussianFiltersize // 2)

        rr = 0.5 * (self.avg_r + self.avg_l) * rr / self.avg_matrix_r
        rr = (rr - 255) * (rr <= 255) + 255
        rr = cv2.GaussianBlur(rr.astype('uint8'), ksize=(self.GaussianFiltersize, self.GaussianFiltersize),
                              sigmaX=self.GaussianFiltersize // 2)

        LRRR_imgs_init = np.stack([self.lr, rr], axis=0).astype("float32")
        LRRR_imgs = (LRRR_imgs_init - np.min(LRRR_imgs_init)) / (np.max(LRRR_imgs_init) - np.min(LRRR_imgs_init))
        self.LRRR_img_torch = self.inputTrans(torch.from_numpy(LRRR_imgs)).unsqueeze(0).cuda()
        output = self.flow_model(self.LRRR_img_torch)
        disp = F.interpolate(output, (self.h, self.w), mode='bicubic', align_corners=False)

        u = disp[0, 0, :, :].detach().cpu().numpy()
        v = disp[0, 1, :, :].detach().cpu().numpy()

        return u, v


'''Calculate strain using gradient kernel filtering'''
def calculate_strain(u, v, filter_size, step_length=1):
    center = [0]
    kernal = []
    for i in range(filter_size // 2):
        m = i + 1
        center.append(1 / (2 * m))
        center.insert(0, -1 / (2 * m))
        kernal.append(center)
        kernal.append(center)
    kernal.append(center)
    avg_num = filter_size * (filter_size - 1) / 2
    kernal_x = (1 / (step_length * avg_num)) * np.array(kernal)
    kernal_y = kernal_x.T
    u_x = cv2.filter2D(u, -1, kernal_x)
    v_y = cv2.filter2D(v, -1, kernal_y)
    u_y = cv2.filter2D(u, -1, kernal_y)
    v_x = cv2.filter2D(v, -1, kernal_x)

    exx = u_x
    eyy = v_y
    exy = 0.5 * (u_y + v_x)

    return exx, eyy, exy



if __name__ == '__main__':
    params_dir = r'F:\case\NetParams\\'
    refimg_dir = r"refimg.bmp"
    curimg_dir = r"curimg.bmp"

    my_calculator = DispCalculator(cropbox=[500, 1012, 500, 1012],      # rectangle region of interest
                                   flow_model_params=params_dir + "flow_best.pth")

    u, v = my_calculator.runcase(lr=cv2.imread(refimg_dir, cv2.IMREAD_GRAYSCALE),
                                 rr=cv2.imread(curimg_dir, cv2.IMREAD_GRAYSCALE))

    # Calculate 2D Strain
    exx, eyy, exy = calculate_strain(u, v, filter_size=7)

    plt.figure(figsize=(12, 8))
    plt.imshow(u)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.imshow(v)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.imshow(exx)
    plt.colorbar()
    plt.show()



