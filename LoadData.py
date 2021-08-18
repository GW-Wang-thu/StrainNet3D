import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class OpticalFlow_Dataset(Dataset):
    '''
    A standard Dataloader using torchvision with 1 channel imgs
    '''
    def __init__(self, data_dir="../Train/", crop=1):
        self.dir = data_dir
        allfiles = os.listdir(data_dir)
        self.imgs = [os.path.join(data_dir, f) for f in allfiles if f.endswith('LRLD_Imgs.npy')]  # input
        self.disp = [os.path.join(data_dir, f) for f in allfiles if f.endswith('LFlow.npy')]  # label
        self.inputTrans = transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))

    def __len__(self):
        return len(self.disp)

    def __getitem__(self, idx):
        input_imgs = np.load(self.imgs[idx])[:, 3:483, 8:648].astype("float32")#[:, :-1, :-1]#[:, 2:-2, 2:-2]
        input_imgs = (input_imgs - np.min(input_imgs)) / (np.max(input_imgs) - np.min(input_imgs))
        output_disps = np.load(self.disp[idx])[:, 3:483, 8:648].astype("float32") #[:, self.crop:-self.crop, self.crop:-self.crop]#
        input_tensor = torch.from_numpy(input_imgs)
        input_tensor = self.inputTrans(input_tensor)
        label_tensor = torch.from_numpy(output_disps)
        return input_tensor, label_tensor


class Disparity_Dataset(Dataset):
    '''
    A standard Dataloader using torchvision with 1 channel imgs
    '''
    def __init__(self, data_dir="../Train/"):
        self.dir = data_dir
        self.allfiles = os.listdir(data_dir)
        self.imgs = [os.path.join(data_dir, f) for f in self.allfiles if f.endswith('LDRD_Imgs.npy')]  # input
        self.disp = [os.path.join(data_dir, f) for f in self.allfiles if f.endswith('Disparity.npy')]  # label
        self.inputTrans = transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        input_imgs_init = np.load(self.imgs[idx])#[:, :-1, :-1]#[:, 2:-2, 2:-2]
        input_imgs = np.zeros(shape=(2, 480, 640)).astype("float32")
        input_imgs[0, :, :] = input_imgs_init[0, 3:483, 8:648]
        input_imgs[1, :, :] = input_imgs_init[1, 3:483, 2:642]

        output_disp_init = np.load(self.disp[idx])  # [:, self.crop-1:-self.crop, self.crop-1:-self.crop] #[:, self.crop:-self.crop, self.crop:-self.crop]#
        output_disp = np.zeros(shape=(1, 480, 640)).astype("float32")
        output_disp[0, :, :] = output_disp_init[0, 3:483, 8:648] + 6.0

        input_imgs = (input_imgs - np.min(input_imgs)) / (np.max(input_imgs) - np.min(input_imgs))
        input_tensor = torch.from_numpy(input_imgs)
        input_tensor = self.inputTrans(input_tensor)

        disp_label_tensor = torch.from_numpy(output_disp)

        return input_tensor, disp_label_tensor


class Fusion_Dataset(Dataset):
    '''
    A standard Dataloader using torchvision with 1 channel imgs
    '''
    def __init__(self, data_dir="../Train/"):
        self.dir = data_dir
        allfiles = os.listdir(data_dir)
        self.LRLD_imgs = [os.path.join(data_dir, f) for f in allfiles if f.endswith('LRLD_Imgs.npy')]  # flowNet input
        self.LDRD_imgs = [os.path.join(data_dir, f) for f in allfiles if f.endswith('LDRD_Imgs.npy')]  # flowNet input
        self.scene_flow = [os.path.join(data_dir, f) for f in allfiles if f.endswith('UVW.npy')]  # label
        self.inputTrans = transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))

    def __len__(self):
        return len(self.scene_flow)

    def __getitem__(self, idx):
        # Flow Images
        LRLD_imgs_init = np.load(self.LRLD_imgs[idx])[:, 3:483, 8:648].astype("float32")  # [:, :-1, :-1]#[:, 2:-2, 2:-2]
        LRLD_imgs = (LRLD_imgs_init - np.min(LRLD_imgs_init)) / (np.max(LRLD_imgs_init) - np.min(LRLD_imgs_init))
        LRLD_tendor = torch.from_numpy(LRLD_imgs)
        LRLD_tendor = self.inputTrans(LRLD_tendor)

        # Disparity Images
        LDRD_imgs_init = np.load(self.LDRD_imgs[idx])
        LDRD_imgs = np.zeros(shape=(2, 480, 640)).astype("float32")
        LDRD_imgs[0, :, :] = LDRD_imgs_init[0, 3:483, 8:648]
        LDRD_imgs[1, :, :] = LDRD_imgs_init[1, 3:483, 2:642]
        LDRD_imgs = (LDRD_imgs - np.min(LDRD_imgs)) / (np.max(LDRD_imgs) - np.min(LDRD_imgs))
        LDRD_tensor = torch.from_numpy(LDRD_imgs)
        LDRD_tensor = self.inputTrans(LDRD_tensor)

        # UVW
        output_disps = np.load(self.scene_flow[idx])[:, 4:482, 9:647].astype("float32")  # [:, self.crop:-self.crop, self.crop:-self.crop]#
        label_tensor = torch.from_numpy(output_disps)

        return LRLD_tendor, LDRD_tensor, label_tensor

