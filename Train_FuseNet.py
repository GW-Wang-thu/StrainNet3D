import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.LoadData import Fusion_Dataset, Fusion_Dataset_1
from Models import FusionNet
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def train():
    epoch_n = 1000
    batchsize = 1
    train_dir = "I:\\DLDIC_3D_Dataset\\Paper_Exp\\train\\"
    test_dir = "I:\\DLDIC_3D_Dataset\\Paper_Exp\\valid\\"
    params_dir = "I:\\DLDIC_3D_Dataset\\Paper_Exp\\parameters\\"
    outputdir = "./Outputs/"
    fuse_lr = 1e-6
    step = 2
    box = [10, 2038, 10, 2038]
    stp = (500, 1000)
    save_raw_uvw = False
    train_cache_dir = "I:\\DLDIC_3D_Dataset\\Paper_Exp\\train\\\disp_flow_cache\\"
    test_cache_dir = "I:\\DLDIC_3D_Dataset\\Paper_Exp\\valid\\disp_flow_cache\\"

    runmode = 1     # 1 - using flow and disparity cache; else - using raw u, v and w cache

    fuse_model = FusionNet.FusionNet_S(params=params_dir + "States_box.json",
                                     cropbox=[stp[0]+box[0], stp[0]+box[1], stp[1]+box[2], stp[1]+box[3]],
                                     transpose_matrix=np.loadtxt(params_dir + "Transpose_matrix.txt", delimiter=',')
                                     ).cuda()

    loss_fn = torch.nn.MSELoss().cuda()
    optimizer_fuse = torch.optim.Adam(fuse_model.parameters(), lr=fuse_lr)
    if runmode == 1:
        train_dl = DataLoader(Fusion_Dataset(data_dir=train_dir, disp_flow_cache_dir=train_cache_dir, box=box),
                              batch_size=batchsize,
                              shuffle=True)

        valid_dl = DataLoader(Fusion_Dataset(data_dir=test_dir, disp_flow_cache_dir=test_cache_dir, box=box),
                              batch_size=batchsize,
                              shuffle=False)
    else:
        train_dl = DataLoader(Fusion_Dataset_1(data_dir=train_dir, disp_flow_cache_dir=train_cache_dir),
                              batch_size=batchsize,
                              shuffle=True)

        valid_dl = DataLoader(Fusion_Dataset_1(data_dir=test_dir, disp_flow_cache_dir=test_cache_dir),
                              batch_size=batchsize,
                              shuffle=False)

    fuse_Recorder = []
    minimum_fuse_loss = 1e5
    now = time.perf_counter()

    if save_raw_uvw:
        step = 1
        fuse_model.eval()

    if os.path.exists(outputdir + "/fuse_last.pth"):
        fuse_Recorder = np.loadtxt(outputdir + "/fuse_Recorder.txt", delimiter=",")
        minimum_fuse_loss = np.min(fuse_Recorder[-50:, 2])
        fuse_Recorder = list(fuse_Recorder)
        fuse_checkpoint = torch.load(outputdir + "/fuse_last.pth")
        fuse_model.load_state_dict(fuse_checkpoint)

    for epoch in range(epoch_n):
        epoch += 1

        if epoch <= (len(fuse_Recorder) - 1) * step:
            continue

        fuse_loss_rec = []

        for i, (idx, tup1, tup2, tup3, tup4) in enumerate(train_dl):
            # flow_input = [tup1[0], tup2[0], tup3[0], tup4[0]]
            inputs = [tup1[0], tup2[0], tup3[0], tup4[0]]
            labels = [tup1[1], tup2[1], tup3[1], tup4[1]]
            # shape = labels[0][0, 0, :, :].shape
            out_cache = []
            for m in range(4):
                if save_raw_uvw:
                    raw_out, fuse_out = fuse_model(flow_output=inputs[m][:, 0:2, :, :],
                                                   disp0=inputs[m][:, 2, :, :].unsqueeze(1),
                                                   disp1=inputs[m][:, 3, :, :].unsqueeze(1))
                else:
                    if runmode == 1:
                        raw_out,  fuse_out = fuse_model(flow_output=inputs[m][:, 0:2, :, :],
                                              disp0=inputs[m][:, 2, :, :].unsqueeze(1),
                                              disp1=inputs[m][:, 3, :, :].unsqueeze(1))       # feed flow and disparity
                    else:
                        raw_out, fuse_out = fuse_model(flow_output=inputs[m][0, 0, :, :],
                                                       disp0=inputs[m][0, 1, :, :],
                                                       disp1=inputs[m][0, 2, :, :], key=1)    # feed raw u, v and w
                    fuse_model.zero_grad()
                    loss_fuse = loss_fn(fuse_out, labels[m])
                    loss_fuse.backward()
                    optimizer_fuse.step()
                    fuse_loss_rec.append(loss_fuse.item())

                if save_raw_uvw:
                    out_cache.append(raw_out[0, :, :, :].cpu().detach().numpy())

            if save_raw_uvw:
                np.save(train_cache_dir + str(int(idx.item())) + "_cache_uvw.npy", np.concatenate(out_cache, axis=0))

        if epoch % step == 0 or epoch == epoch_n - 1:

            train_fuse_loss_mean = np.mean(np.array(fuse_loss_rec), axis=0)
            valid_fuse_loss_rec = []
            valid_fuse_raw_loss_rec = []

            for i, (idx, tup1, tup2, tup3, tup4) in enumerate(valid_dl):
                inputs = [tup1[0], tup2[0], tup3[0], tup4[0]]
                labels = [tup1[1], tup2[1], tup3[1], tup4[1]]
                if int(idx.item()) == 10:
                    pass
                out_cache = []
                for m in range(4):
                    if save_raw_uvw:
                        raw_out, fuse_out = fuse_model(flow_output=inputs[m][:, 0:2, :, :],
                                                       disp0=inputs[m][:, 2, :, :].unsqueeze(1),
                                                       disp1=inputs[m][:, 3, :, :].unsqueeze(1))
                    else:

                        if runmode == 1:
                            raw_out, fuse_out = fuse_model(flow_output=inputs[m][:, 0:2, :, :],
                                                           disp0=inputs[m][:, 2, :, :].unsqueeze(1),
                                                           disp1=inputs[m][:, 3, :, :].unsqueeze(1))
                        else:
                            raw_out, fuse_out = fuse_model(flow_output=inputs[m][0, 0, :, :],
                                                           disp0=inputs[m][0, 1, :, :],
                                                           disp1=inputs[m][0, 2, :, :], key=1)
                        loss_fuse_out = loss_fn(fuse_out, labels[m])
                        loss_fuse_raw = loss_fn(raw_out, labels[m])
                        valid_fuse_loss_rec.append(loss_fuse_out.item())
                        valid_fuse_raw_loss_rec.append(loss_fuse_raw.item())

                    if save_raw_uvw:
                        out_cache.append(raw_out[0, :, :, :].cpu().detach().numpy())

                if save_raw_uvw:  # and not os.path.exists(train_cache_dir + str(int(idx.item())) + "_cache_flow.npy"):
                    np.save(test_cache_dir + str(int(idx.item())) + "_cache_uvw.npy", np.concatenate(out_cache, axis=0))

            valid_fuse_mean_loss = np.mean(np.array(valid_fuse_loss_rec))
            valid_fuse_raw_mean_loss = np.mean(np.array(valid_fuse_raw_loss_rec))
            print("Epoch %d," % ((len(fuse_Recorder) - 1) * step),
                  " train flow loss: ", train_fuse_loss_mean,
                  ", valid flow loss: ", valid_fuse_mean_loss,
                  ", valid raw flow loss: ", valid_fuse_raw_mean_loss,
                  ", timeconsume %f" % (time.perf_counter() - now))
            now = time.perf_counter()
            fuse_Recorder.append([fuse_lr, train_fuse_loss_mean, valid_fuse_mean_loss])
            if (valid_fuse_mean_loss < minimum_fuse_loss):
                minimum_fuse_loss = valid_fuse_mean_loss
                torch.save(fuse_model.state_dict(), outputdir + "/" + "fuse_best.pth")

            torch.save(fuse_model.state_dict(), outputdir + "/fuse_last.pth")
            np.savetxt(outputdir + "/fuse_Recorder.txt", np.array(fuse_Recorder), delimiter=",")


if __name__ == '__main__':
    train()