import numpy as np
import torch
from torch.utils.data import DataLoader
from LoadData import OpticalFlow_Dataset
from Models import FlowNet as flowNet
import os
import time


def train():
    epoch_n = 10000
    batchsize = 4
    train_dir = "../dataset\\Train/"
    valid_dir = "../dataset\\Valid/"
    outputdir = "./Outputs/"
    flow_lr = 1e-6
    step=2

    flow_model = flowNet.DICNN().cuda()

    optimizer_flow = torch.optim.Adam(flow_model.parameters(), lr=flow_lr)

    train_dl = DataLoader(OpticalFlow_Dataset(train_dir),
                          batch_size=batchsize,
                          shuffle=True)

    valid_dl = DataLoader(OpticalFlow_Dataset(valid_dir),
                          batch_size=batchsize,
                          shuffle=True)

    flow_Recorder = []

    minimum_flow_loss = 1e5

    if os.path.exists(outputdir + "/flow_last.pth"):
        flow_Recorder = np.loadtxt(outputdir + "/flow_Recorder.txt", delimiter=",")
        minimum_flow_loss = np.min(flow_Recorder[-50:, 1])
        flow_Recorder = list(flow_Recorder)
        flow_checkpoint = torch.load(outputdir + "/flow_last.pth")
        flow_model.load_state_dict(flow_checkpoint)

    now = time.perf_counter()

    weights = [1, 0, 0]
    for epoch in range(epoch_n):
        flow_model.train()
        flow_loss_rec = []

        if (epoch > 0) and (epoch % 100 == 0):
            weights = [0.5, 0.25, 0.25]

        if epoch+1 % 1000 == 0:
            optimizer_flow = torch.optim.Adam(flow_model.parameters(), lr=flow_lr * 0.5)

        if epoch < len(flow_Recorder):
            continue

        for i, (train_batch, label_batch) in enumerate(train_dl):
            train_batch, flow_label_batch = train_batch.cuda(), label_batch.cuda()
            flow_output, pr1, pr2 = flow_model(train_batch)

            flow_model.zero_grad()
            loss_flow = flowNet.loss_flow([flow_output, pr1, pr2], flow_label_batch, weights)

            loss_flow[3].backward()
            optimizer_flow.step()

            flow_loss_rec.append([loss.item() for loss in loss_flow])


        if epoch % step == 0 or epoch == epoch_n - 1:
            train_flow_loss_mean = np.mean(np.array(flow_loss_rec), axis=0)

            valid_flow_loss_rec = []
            for i, (valid_batch, label_batch) in enumerate(valid_dl):
                valid_batch, flow_label_batch = valid_batch.cuda(0), label_batch.cuda(0)
                flow_output, pr1, pr2 = flow_model(valid_batch)
                loss_flow = flowNet.loss_flow([flow_output, pr1, pr2], flow_label_batch, weights)
                valid_flow_loss_rec.append([loss.item() for loss in loss_flow])

            valid_flow_loss_mean = np.mean(np.array(valid_flow_loss_rec), axis=0)

            print("Epoch %d," % ((len(flow_Recorder) - 1) * step), " train disp loss: ", train_flow_loss_mean[:],
                  ", valid disp loss: ", valid_flow_loss_mean[:],
                  ", timeconsume %f" % (time.perf_counter() - now))
            now = time.perf_counter()

            flow_Recorder.append([flow_lr] + list(train_flow_loss_mean[:]) + list(valid_flow_loss_mean[:]))

            if (valid_flow_loss_mean[-1] < minimum_flow_loss):
                minimum_flow_loss = valid_flow_loss_mean[-1]
                torch.save(flow_model.state_dict(), outputdir + "/"+"flow_best.pth")

            torch.save(flow_model.state_dict(), outputdir + "/flow_last.pth")
            np.savetxt(outputdir + "/flow_Recorder.txt", np.array(flow_Recorder), delimiter=",")


if __name__ == '__main__':
    train()
