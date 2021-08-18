import numpy as np
import cv2
import os
import re


def gen_dataset(in_dir, out_dir, train_percent, start_idx=300):
    filenames = os.listdir(in_dir)
    filenames = [os.path.join(in_dir, f) for f in filenames if f.endswith('LR.tif')]
    for i in range(len(filenames)):
        print('\r', '%d of %d finished ' % (i, len(filenames)), end='\b')
        if i < start_idx:
            continue
        proj_name = filenames[i][-11:]
        k = re.sub("\D", "", proj_name)
        LR_img = cv2.imread(in_dir + str(k) + "_LR.tif", cv2.IMREAD_GRAYSCALE)
        LD_img = cv2.imread(in_dir + str(k) + "_LD.tif", cv2.IMREAD_GRAYSCALE)
        RR_img = cv2.imread(in_dir + str(k) + "_RR.tif", cv2.IMREAD_GRAYSCALE)
        RD_img = cv2.imread(in_dir + str(k) + "_RD.tif", cv2.IMREAD_GRAYSCALE)
        Disp_X = np.loadtxt(in_dir + str(k) + "_Disparity_DX.csv")
        Disp_Y = np.loadtxt(in_dir + str(k) + "_Disparity_DY.csv")
        Flow_X = np.loadtxt(in_dir + str(k) + "LFU.csv")
        Flow_Y = np.loadtxt(in_dir + str(k) + "LFV.csv")
        LUVW_U = np.loadtxt(in_dir + str(k) + "_LWU.csv")
        LUVW_V = np.loadtxt(in_dir + str(k) + "_LWV.csv")
        LUVW_W = np.loadtxt(in_dir + str(k) + "_LWW.csv")

        disp_stackedimg = np.stack([LD_img, RD_img], axis=0)
        flow_stackedimg = np.stack([LR_img, LD_img], axis=0)

        disparity_stacked = np.stack([Disp_X, Disp_Y], axis=0)
        opticalflow_stacked = np.stack([Flow_X, Flow_Y], axis=0)
        LUVW_stacked = np.stack([LUVW_U, LUVW_V, LUVW_W], axis=0)

        np.random.seed(i)
        if np.random.rand() < train_percent:
            output_dir = out_dir + "Train/"
        else:
            output_dir = out_dir + "Valid/"

        randomnum = np.random.randint(1, 10) * 1000000 + 30000000
        np.save(output_dir + str(randomnum + i * 2 + 1 + start_idx) + "_LDRD_Imgs.npy", disp_stackedimg)
        np.save(output_dir + str(randomnum + i * 2 + 1 + start_idx) + "_LRLD_Imgs.npy", flow_stackedimg)
        np.save(output_dir + str(randomnum + i * 2 + 1 + start_idx) + "_Disparity.npy", disparity_stacked)
        np.save(output_dir + str(randomnum + i * 2 + 1 + start_idx) + "_LFlow.npy", opticalflow_stacked)
        np.save(output_dir + str(randomnum + i * 2 + 1 + start_idx) + "_UVW.npy", LUVW_stacked)



if __name__ == '__main__':
    gen_dataset(in_dir="D:\\Guowen\\DLDIC_3D\\dataset\\data3\\", out_dir="D:\\Guowen\\DLDIC_3D\\dataset\\", train_percent=0.85, start_idx=0)
