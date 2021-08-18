import numpy as np
import torch
from torch.utils.data import DataLoader
from LoadData import Disparity_Dataset
from Models import DispNet
import os
import time


def train():
    epoch_n = 10000
    batchsize = 8
    train_dir = "../dataset\\Train/"
    valid_dir = "../dataset\\Valid/"
    outputdir = "./Outputs/"
    disp_lr = 1e-6
    step=2

    disp_model = DispNet.DispNet().cuda()

    optimizer_disp = torch.optim.Adam(disp_model.parameters(), lr=disp_lr)

    train_dl = DataLoader(Disparity_Dataset(train_dir),
                          batch_size=batchsize,
                          shuffle=True)

    valid_dl = DataLoader(Disparity_Dataset(valid_dir),
                          batch_size=batchsize,
                          shuffle=True)

    disp_Recorder = []

    minimum_disp_loss = 1e5

    if os.path.exists(outputdir + "/disp_last.pth"):
        disp_Recorder = np.loadtxt(outputdir + "/disp_Recorder.txt", delimiter=",")
        minimum_disp_loss = np.min(disp_Recorder[-50:, 1])
        disp_Recorder = list(disp_Recorder)
        disp_checkpoint = torch.load(outputdir + "/disp_last.pth")
        disp_model.load_state_dict(disp_checkpoint)

    now = time.perf_counter()

    weights = [1, 0, 0, 0, 0]
    for epoch in range(epoch_n):
        disp_model.train()
        disp_loss_rec = []
        mean_loss_rec = []

        if (epoch > 0) and (epoch % 100 == 0):
            weights = [0.4, 0.25, 0.15, 0.1, 0.1]
            optimizer_disp = torch.optim.Adam(disp_model.parameters(), lr=disp_lr * 2)
        if epoch+1 % 1000 == 0:
            optimizer_disp = torch.optim.Adam(disp_model.parameters(), lr=disp_lr * 0.5)

        if epoch < len(disp_Recorder):
            continue

        for i, (train_batch, label_batch) in enumerate(train_dl):
            train_batch, disp_label_batch = train_batch.cuda(), label_batch.cuda()
            disp_output, pr2, pr3, pr4, pr5 = disp_model(train_batch)

            disp_model.zero_grad()
            loss_disp, total_loss = DispNet.lossEPE([disp_output,  pr2, pr3, pr4, pr5], weights, disp_label_batch)

            total_loss.backward()
            optimizer_disp.step()

            disp_loss_rec.append([loss.item() for loss in loss_disp])
            mean_loss_rec.append(total_loss.item())

        if epoch % step == 0 or epoch == epoch_n - 1:
            train_disp_all_loss_mean = np.mean(np.array(disp_loss_rec), axis=0)

            valid_disp_loss_rec = []
            valid_all_loss_rec = []
            for i, (valid_batch, label_batch) in enumerate(valid_dl):
                valid_batch, disp_label_batch = valid_batch.cuda(0), label_batch.cuda(0)
                disp_output, pr2, pr3, pr4, pr5 = disp_model(valid_batch)

                loss_disp, total_loss = DispNet.lossEPE([disp_output, pr2, pr3, pr4, pr5], weights, disp_label_batch)

                valid_disp_loss_rec.append(total_loss.item())
                valid_all_loss_rec.append([loss.item() for loss in loss_disp])

            valid_disp_all_loss_mean = np.mean(np.array(valid_all_loss_rec), axis=0)

            print("Epoch %d,"% ((len(disp_Recorder) - 1) * step),  " train disp loss: ", train_disp_all_loss_mean[0:6], ", valid disp loss: ", valid_disp_all_loss_mean[0:6], ", timeconsume %f"%(time.perf_counter() - now))
            now = time.perf_counter()

            disp_Recorder.append([disp_lr] + list(train_disp_all_loss_mean[:]) + list(valid_disp_all_loss_mean[:]))

            if (valid_disp_all_loss_mean[5] < minimum_disp_loss):
                minimum_disp_loss = valid_disp_all_loss_mean[5]
                torch.save(disp_model.state_dict(), outputdir + "/"+"disp_best.pth")

            torch.save(disp_model.state_dict(), outputdir + "/disp_last.pth")
            np.savetxt(outputdir + "/disp_Recorder.txt", np.array(disp_Recorder), delimiter=",")


if __name__ == '__main__':
    train()

