import numpy as np
import torch
from torch.utils.data import DataLoader
from LoadData import Fusion_Dataset
from Models import DispNet, FlowNet, FuseNet
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt


def speed_test():
    epoch_n = 10000
    batchsize = 16
    train_dir = "../dataset\\Train/"
    coord_dir = "../dataset\\coordinates/"
    outputdir = "./Outputs1/"
    fuse_lr = 1e-8
    step = 1

    disp_model = DispNet.DispNet().cuda()
    flow_model = FlowNet.DICNN().cuda()
    fuse_model = FuseNet.FusionNet().cuda()

    fuse_train_dl = DataLoader(Fusion_Dataset(train_dir), batch_size=batchsize, shuffle=True)
    coord_x = np.loadtxt(coord_dir+"Disparity_RX.csv")[3:483, 8:648].astype("float32")
    coord_x = F.interpolate(torch.from_numpy(coord_x).unsqueeze(0).unsqueeze(0).cuda(), (240, 320), mode='bicubic', align_corners=False)

    disp_params = torch.load(outputdir + "/disp_best.pth")
    disp_model.load_state_dict(disp_params)
    flow_params = torch.load(outputdir + "/flow_best.pth")
    flow_model.load_state_dict(flow_params)

    if os.path.exists(outputdir + "/fuse_best.pth"):
        fuse_checkpoint = torch.load(outputdir + "/fuse_best.pth")
        fuse_model.load_state_dict(fuse_checkpoint)

    for epoch in range(epoch_n):
        fuse_model.eval()
        disp_model.eval()
        flow_model.eval()

        for i, (flow_batch, disp_batch, label_batch) in enumerate(fuse_train_dl):
            flow_batch, disp_batch, label_batch = flow_batch.cuda(), disp_batch.cuda(), label_batch.cuda()

            time0 = time.perf_counter()
            time1 = time.perf_counter()
            print("********==========********")
            flow_output_batch = flow_model(flow_batch)
            print("flow_time: ", time.perf_counter()-time1)
            time1 = time.perf_counter()
            disp_output_batch = disp_model(disp_batch) - 6.0
            print("Disp_time: ", time.perf_counter()-time1)
            time1 = time.perf_counter()
            batch, _, _, _ = flow_output_batch.size()
            stacked_input = torch.cat((flow_output_batch, disp_output_batch, coord_x.expand(batch, 1, 240, 320)), dim=1)
            fuse_output_batch = fuse_model(stacked_input)
            print("Fuse_time: ", time.perf_counter()-time1)
            print("ALL Time: ", time.perf_counter()-time0)
            print("batch size: ", batch)


if __name__ == '__main__':
    speed_test()