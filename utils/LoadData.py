import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt


class OpticalFlow_Dataset(Dataset):
    '''
    A standard Dataloader
    '''

    def __init__(self, data_dir="../Train/", cropsize=(512, 640), box=(32, 1820, 0, 2685), random_crop=False):
        self.cropsize = cropsize
        self.dir = data_dir
        self.id_list = np.loadtxt(data_dir + "idlist.csv", delimiter=",", dtype="int32").tolist()   # "idlist.csv" must be put in the dataset directory".
        self.box = box                                 # Only image blocks within the box are randomly-cropped and used.
        self.random_crop = random_crop
        self.blur3 = transforms.GaussianBlur(3, 1.5)
        self.blur5 = transforms.GaussianBlur(5, 3)
        self.blur7 = transforms.GaussianBlur(7, 4)     # Perform Gaussian filter on the images using random-size kernel.

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, id):
        # L0, L1, L2, L3, R0, R1, R2, R3
        idx = self.id_list[id]
        temp_img = np.load(self.dir + str(idx) + "_imgs.npy")
        temp_flow = np.load(self.dir + str(idx) + "_flow.npy")      # load 4 pairs of a dataset idx at one time
        #
        if self.random_crop:
            a = np.random.randint(20, 31)
            b = np.random.randint(20, 31)
            self.cropsize = (a * 64, b * 64)                        # image size can must be divided by 32 (2^5)

        try:
            start_x = np.random.randint(self.box[0], self.box[1]-self.cropsize[0])
            start_y = np.random.randint(self.box[2], self.box[3]-self.cropsize[1])
        except:
            start_x = 0
            start_y = 0

        temp_flow = torch.from_numpy(temp_flow.astype("float32")).cuda()
        temp_img = torch.from_numpy(temp_img.astype("float32")).cuda()

        se = np.random.rand()           # random filter size
        if se < 0.2:
            temp_img = self.blur3(temp_img)
        elif se < 0.5:
            temp_img = self.blur5(temp_img)
        elif se < 0.8:
            temp_img = self.blur7(temp_img)

        '''image normalization'''
        min_gray = torch.min(temp_img[0, 0:100, 0:100])
        max_gray = torch.max(temp_img[0, 0:100, 0:100])
        ratio = 1.0 / (max_gray - min_gray)
        temp_tensor = (temp_img - min_gray) * ratio - 0.50

        # load 4 pairs of a dataset idx at one time (scene pairs: 0-1, 1-2, 1-3, 2-3)
        input_01, input_12, input_13, input_23 = torch.stack([temp_tensor[0, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], temp_tensor[1, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]], dim=0), \
                                                 torch.stack([temp_tensor[1, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], temp_tensor[2, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]], dim=0), \
                                                 torch.stack([temp_tensor[1, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], temp_tensor[3, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]], dim=0), \
                                                 torch.stack([temp_tensor[2, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], temp_tensor[3, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]], dim=0)

        return idx,\
               (input_01, temp_flow[0:2, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]), \
               (input_12, temp_flow[2:4, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]), \
               (input_13, temp_flow[4:6, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]), \
               (input_23, temp_flow[6:8, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]])


class Disparity_Dataset(Dataset):
    '''
    A standard Dataloader
    '''
    def __init__(self, data_dir, cropsize, box, random_crop=False):
        self.dir = data_dir
        self.id_list = np.loadtxt(data_dir + "idlist.csv", delimiter=",", dtype="int32",encoding='UTF-8').tolist()
        self.cropsize = cropsize
        self.box = box
        self.random_crop = random_crop
        self.plane_disparity = torch.from_numpy(np.zeros(shape=cropsize).astype("float32")).cuda()
        self.blur3 = transforms.GaussianBlur(3, 1.5)
        self.blur5 = transforms.GaussianBlur(5, 3)
        self.blur7 = transforms.GaussianBlur(7, 4)      # Perform Gaussian filter on the images using random-size kernel

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, id):
        idx = self.id_list[id]
        temp_img = np.load(self.dir + str(idx) + "_imgs.npy")
        temp_disp = np.load(self.dir + str(idx) + "_disp.npy")

        if self.random_crop:
            a = np.random.randint(20, 31)
            b = np.random.randint(20, 31)
            self.cropsize = (a * 64, b * 64)
        try:
            start_x = np.random.randint(self.box[0], self.box[1] - self.cropsize[0])
            start_y = np.random.randint(self.box[2], self.box[3] - self.cropsize[1])
        except:
            start_x = 0
            start_y = 0

        temp_disp = torch.from_numpy(temp_disp.astype("float32")).cuda()

        temp_img = torch.from_numpy(temp_img.astype("float32")).cuda()

        se = np.random.rand()
        if se < 0.2:
            temp_img = self.blur3(temp_img)
        elif se < 0.5:
            temp_img = self.blur5(temp_img)
        elif se < 0.8:
            temp_img = self.blur7(temp_img)

        min_gray = torch.min(temp_img[0, 0:100, 0:100])
        max_gray = torch.max(temp_img[0, 0:100, 0:100])
        ratio = 1.0 / (max_gray - min_gray)
        temp_tensor = (temp_img - min_gray) * ratio - 0.50

        input_0, input_1, input_2, input_3 = torch.stack([temp_tensor[0, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], temp_tensor[4, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]], dim=0), \
                                             torch.stack([temp_tensor[1, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], temp_tensor[5, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]], dim=0), \
                                             torch.stack([temp_tensor[2, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], temp_tensor[6, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]], dim=0), \
                                             torch.stack([temp_tensor[3, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], temp_tensor[7, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]]], dim=0)

        disp_0, disp_1, disp_2, disp_3 = torch.unsqueeze(self.plane_disparity, 0),\
                                         torch.unsqueeze(temp_disp[0, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], 0), \
                                         torch.unsqueeze(temp_disp[1, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], 0), \
                                         torch.unsqueeze(temp_disp[2, start_x:start_x+self.cropsize[0], start_y:start_y+self.cropsize[1]], 0),
        # plt.imshow(input_1[0, :, :].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(input_1[1, :, :].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(disp_1[0, :, :].cpu().detach().numpy())
        # plt.show()

        return idx,\
               (input_0, disp_0), \
               (input_1, disp_1), \
               (input_2, disp_2), \
               (input_3, disp_3)


'''Dataloader to train DispRefineNet'''
class Fusion_Dataset(Dataset):
    '''Version 1: Load the saved disparity and optical flow cache file calculated using SubpixelCorrNet'''
    def __init__(self, data_dir="../Train/", disp_flow_cache_dir="../Train/disp_flow_cache/", box=[10, 2038, 10, 2038]):
        self.dir = data_dir
        self.box = box
        self.disp_flow_cache_dir = disp_flow_cache_dir
        self.id_list = np.loadtxt(data_dir + "idlist.csv", delimiter=",", dtype="int32").tolist()

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, id):
        # input: flowX, flowY, disp0, disp1
        idx = self.id_list[id]
        temp_flow_cache = torch.from_numpy(np.load(self.disp_flow_cache_dir + str(idx) + "_cache_flow.npy").astype("float32")).cuda()
        temp_disp_cache = torch.from_numpy(np.load(self.disp_flow_cache_dir + str(idx) + "_cache_disp.npy").astype("float32")).cuda()
        temp_uvw = torch.from_numpy(np.load(self.dir + str(idx) + "_uvws.npy").astype("float32")).cuda()

        input_01, input_12, input_13, input_23 = torch.cat([temp_flow_cache[0:2, self.box[0]:self.box[1], self.box[2]:self.box[3]], temp_disp_cache[0, self.box[0]:self.box[1], self.box[2]:self.box[3]].unsqueeze(0), temp_disp_cache[1, self.box[0]:self.box[1], self.box[2]:self.box[3]].unsqueeze(0)], dim=0), \
                                                 torch.cat([temp_flow_cache[2:4, self.box[0]:self.box[1], self.box[2]:self.box[3]], temp_disp_cache[1, self.box[0]:self.box[1], self.box[2]:self.box[3]].unsqueeze(0), temp_disp_cache[2, self.box[0]:self.box[1], self.box[2]:self.box[3]].unsqueeze(0)], dim=0), \
                                                 torch.cat([temp_flow_cache[4:6, self.box[0]:self.box[1], self.box[2]:self.box[3]], temp_disp_cache[1, self.box[0]:self.box[1], self.box[2]:self.box[3]].unsqueeze(0), temp_disp_cache[3, self.box[0]:self.box[1], self.box[2]:self.box[3]].unsqueeze(0)], dim=0), \
                                                 torch.cat([temp_flow_cache[6:8, self.box[0]:self.box[1], self.box[2]:self.box[3]], temp_disp_cache[2, self.box[0]:self.box[1], self.box[2]:self.box[3]].unsqueeze(0), temp_disp_cache[3, self.box[0]:self.box[1], self.box[2]:self.box[3]].unsqueeze(0)], dim=0)

        uvw_01, uvw_12, uvw_13, uvw_23 = temp_uvw[0:3, self.box[0]:self.box[1], self.box[2]:self.box[3]], temp_uvw[3:6, self.box[0]:self.box[1], self.box[2]:self.box[3]], \
                                         temp_uvw[6:9, self.box[0]:self.box[1], self.box[2]:self.box[3]], temp_uvw[9:12, self.box[0]:self.box[1], self.box[2]:self.box[3]]

        return idx, \
               (input_01, uvw_01), \
               (input_12, uvw_12), \
               (input_13, uvw_13), \
               (input_23, uvw_23)


class Fusion_Dataset_1(Dataset):
    '''Version 2: Load the theoretically calculated u, v and w cache file.'''
    def __init__(self, data_dir="../Train/", disp_flow_cache_dir="../Train/disp_flow_cache/"):
        self.dir = data_dir
        self.disp_flow_cache_dir = disp_flow_cache_dir
        self.id_list = np.loadtxt(data_dir + "idlist.csv", delimiter=",", dtype="int32").tolist()

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, id):
        idx = self.id_list[id]
        self.box = [0, 2028, 0, 2028]
        # print(idx)
        temp_flow_cache = torch.from_numpy(np.load(self.disp_flow_cache_dir + str(idx) + "_cache_uvw.npy").astype("float32")).cuda()
        temp_uvw = torch.from_numpy(np.load(self.dir + str(idx) + "_uvws.npy").astype("float32")).cuda()

        input_01, input_12, input_13, input_23 = temp_flow_cache[0:3, self.box[0]:self.box[1], self.box[2]:self.box[3]], \
                                                 temp_flow_cache[3:6, self.box[0]:self.box[1], self.box[2]:self.box[3]], \
                                                 temp_flow_cache[6:9, self.box[0]:self.box[1], self.box[2]:self.box[3]], \
                                                 temp_flow_cache[9:12, self.box[0]:self.box[1], self.box[2]:self.box[3]]

        uvw_01, uvw_12, uvw_13, uvw_23 = temp_uvw[0:3, 10+self.box[0]:10+self.box[1], 10+self.box[2]:10+self.box[3]], temp_uvw[3:6, 10+self.box[0]:10+self.box[1], 10+self.box[2]:10+self.box[3]], \
                                         temp_uvw[6:9, 10+self.box[0]:10+self.box[1], 10+self.box[2]:10+self.box[3]], temp_uvw[9:12, 10+self.box[0]:10+self.box[1], 10+self.box[2]:10+self.box[3]]

        return idx, \
               (input_01, uvw_01), \
               (input_12, uvw_12), \
               (input_13, uvw_13), \
               (input_23, uvw_23)