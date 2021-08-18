import numpy as np
import torch
from torch.utils.data import DataLoader
from LoadData import Fusion_Dataset
from Models import DispNet, FlowNet, FuseNet
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def train():
    epoch_n = 8000
    batchsize = 8
    train_dir = "../dataset\\Train/"
    valid_dir = "../dataset\\Valid/"
    coord_dir = "../dataset\\coordinates/"
    outputdir = "./Outputs1/"
    fuse_lr = 1e-6
    step = 4

    disp_model = DispNet.DispNet().cuda()
    flow_model = FlowNet.DICNN().cuda()
    fuse_model = FuseNet.FusionNet().cuda()

    loss_fn = torch.nn.MSELoss().cuda()
    optimizer_fuse = torch.optim.Adam(fuse_model.parameters(), lr=fuse_lr)

    fuse_train_dl = DataLoader(Fusion_Dataset(train_dir), batch_size=batchsize, shuffle=True)
    fuse_valid_dl = DataLoader(Fusion_Dataset(valid_dir), batch_size=batchsize, shuffle=True)

    coord_x = np.loadtxt(coord_dir+"Disparity_RX.csv")[3:483, 8:648].astype("float32")
    coord_x = F.interpolate(torch.from_numpy(coord_x).unsqueeze(0).unsqueeze(0).cuda(), (240, 320), mode='bicubic', align_corners=False)

    fuse_Recorder = []
    minimum_fuse_loss = 1e5
    now = time.perf_counter()

    disp_params = torch.load(outputdir + "/disp_best.pth")
    disp_model.load_state_dict(disp_params)
    flow_params = torch.load(outputdir + "/flow_best.pth")
    flow_model.load_state_dict(flow_params)
    # torch.save(disp_model.state_dict(), outputdir + "/disp_best_v0.pth",  _use_new_zipfile_serialization=False)   # Save model for old version of pytorch
    # torch.save(flow_model.state_dict(), outputdir + "/flow_best_v0.pth",  _use_new_zipfile_serialization=False)

    if os.path.exists(outputdir + "/fuse_last.pth"):
        fuse_Recorder = np.loadtxt(outputdir + "/fuse_Recorder.txt", delimiter=",")
        minimum_fuse_loss = np.min(fuse_Recorder[-50:, 2])
        fuse_Recorder = list(fuse_Recorder)
        fuse_checkpoint = torch.load(outputdir + "/fuse_last.pth")
        fuse_model.load_state_dict(fuse_checkpoint)
        # torch.save(fuse_model.state_dict(), outputdir + "/fuse_best_v0.pth", _use_new_zipfile_serialization=False)

    for epoch in range(epoch_n):
        fuse_model.train()
        disp_model.eval()
        flow_model.eval()

        # if (epoch+1) % 1000 == 0:
        #     optimizer_fuse = torch.optim.Adam(flow_model.parameters(), lr=fuse_lr * 0.5)

        if epoch // step < len(fuse_Recorder):
            continue

        fuse_loss_rec = []

        for i, (flow_batch, disp_batch, label_batch) in enumerate(fuse_train_dl):
            flow_batch, disp_batch, label_batch = flow_batch.cuda(), disp_batch.cuda(), label_batch.cuda()

            flow_output_batch = flow_model(flow_batch)
            disp_output_batch = disp_model(disp_batch) - 6.0

            batch, _, _, _ = flow_output_batch.size()
            stacked_input = torch.cat((flow_output_batch, disp_output_batch, coord_x.expand(batch, 1, 240, 320)), dim=1)
            fuse_output_batch = fuse_model(stacked_input)

            fuse_model.zero_grad()

            loss_fuse = loss_fn(F.interpolate(fuse_output_batch, (480, 640), mode='bicubic', align_corners=False)[:, :, 1:-1, 1:-1], label_batch)

            loss_fuse.backward()

            optimizer_fuse.step()

            fuse_loss_rec.append(loss_fuse.item())

        if epoch % step == 0 or epoch == epoch_n - 1:

            train_fuse_loss_mean = np.mean(np.array(fuse_loss_rec), axis=0)
            valid_fuse_loss_rec = []

            for i, (flow_batch, disp_batch, label_batch) in enumerate(fuse_valid_dl):
                flow_batch, disp_batch, label_batch = flow_batch.cuda(), disp_batch.cuda(), label_batch.cuda()

                flow_output_batch = flow_model(flow_batch)
                disp_output_batch = disp_model(disp_batch) - 6.0

                batch, _, _, _ = flow_output_batch.size()
                stacked_input = torch.cat((flow_output_batch, disp_output_batch, coord_x.expand(batch, 1, 240, 320)), dim=1)
                fuse_output_batch = fuse_model(stacked_input)

                loss_fuse = loss_fn(F.interpolate(fuse_output_batch, (480, 640), mode='bicubic', align_corners=False)[:, :, 1:-1, 1:-1], label_batch)

                valid_fuse_loss_rec.append(loss_fuse.item())

            valid_fuse_mean_loss = np.mean(np.array(valid_fuse_loss_rec))

            print("Epoch %d," % ((len(fuse_Recorder) - 1) * step), " train flow loss: ", train_fuse_loss_mean,
                  ", valid flow loss: ", valid_fuse_mean_loss, ", timeconsume %f" % (time.perf_counter() - now))
            now = time.perf_counter()

            fuse_Recorder.append([fuse_lr, train_fuse_loss_mean, valid_fuse_mean_loss])
            if (valid_fuse_mean_loss < minimum_fuse_loss):
                minimum_fuse_loss = valid_fuse_mean_loss
                torch.save(fuse_model.state_dict(), outputdir + "/" + "fuse_best.pth")

            torch.save(fuse_model.state_dict(), outputdir + "/fuse_last.pth")
            np.savetxt(outputdir + "/fuse_Recorder.txt", np.array(fuse_Recorder), delimiter=",")


if __name__ == '__main__':
    train()