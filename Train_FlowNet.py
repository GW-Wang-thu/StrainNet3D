import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.LoadData import OpticalFlow_Dataset, Disparity_Dataset
from Models import SubpixelCorrNet
import os
import time
import torch.nn.functional as F


def train():
    epoch_n = 4000
    batchsize = 1
    train_dir = "I:\\DLDIC_3D_Dataset\\Paper_Exp\\train\\"
    valid_dir = "I:\\DLDIC_3D_Dataset\\Paper_Exp\\valid\\"
    outputdir = "./Outputs/"
    flow_lr = 1e-4
    step = 1
    option="flow"
    random_crop = False
    cropsize=(1984, 1984)
    box_flow=(10, 2035, 10, 2035)
    box_disp=(0, 2048, 0, 2048)
    save_results = False
    if save_results:    # Save full-size results to train DispRefineNet.
        step = 1
        random_crop = False
        cropsize=(2048, 2048)
        box_flow = (0, 2048, 0, 2048)
        box_disp=(0, 2048, 0, 2048)
    train_cache_dir = "I:\\DLDIC_3D_Dataset\\Paper_Exp\\train\\\disp_flow_cache\\"
    test_cache_dir = "I:\\DLDIC_3D_Dataset\\Paper_Exp\\valid\\disp_flow_cache\\"

    flow_model = SubpixelCorrNet.SubpixelCorrNet().cuda()

    optimizer_flow = torch.optim.Adam(flow_model.parameters(), lr=flow_lr)

    train_dl_flow = DataLoader(OpticalFlow_Dataset(train_dir, cropsize=cropsize, box=box_flow, random_crop=random_crop),
                          batch_size=batchsize,
                          shuffle=True)

    valid_dl_flow = DataLoader(OpticalFlow_Dataset(valid_dir, cropsize=cropsize, box=box_flow, random_crop=random_crop),
                          batch_size=batchsize,
                          shuffle=True)

    train_dl_disp = DataLoader(Disparity_Dataset(train_dir, cropsize=cropsize, box=box_disp, random_crop=random_crop),
                          batch_size=batchsize,
                          shuffle=True)

    valid_dl_disp = DataLoader(Disparity_Dataset(valid_dir, cropsize=cropsize, box=box_disp, random_crop=random_crop),
                          batch_size=batchsize,
                          shuffle=True)
    flow_Recorder = []

    minimum_flow_loss = 1e5

    if os.path.exists(outputdir + "/"+option+"_last.pth"):      # Load pre-trained.
        flow_Recorder = np.loadtxt(outputdir + "/"+option+"_Recorder.txt", delimiter=",")
        minimum_flow_loss = np.min(flow_Recorder[:, -1])
        flow_Recorder = list(flow_Recorder)
        flow_checkpoint = torch.load(outputdir + "/"+option+"_last.pth")
        flow_model.load_state_dict(flow_checkpoint)

    now = time.perf_counter()

    weights = [1, 0, 0]         # weights of loss of different zoom level.
    for epoch in range(epoch_n):
        epoch += 1
        flow_model.train()
        if save_results:
            flow_model.eval()
        flow_loss_rec = []
        disp_loss_rec = []

        if epoch <= (len(flow_Recorder) - 1) * step:
            continue

        if (epoch > 0) and (epoch % 5 == 1):    # Alternate loss weights
            weights = [0.5, 0.3, 0.2]
        else:
            weights = [1, 0, 0]

        '''Train'''
        if option == "flow":

            for i, (idx, tup01, tup12, tup13, tup23) in enumerate(train_dl_flow):
                inputs = [tup01[0], tup12[0], tup13[0], tup23[0]]       # 4 image pairs of 1 dataset
                labels = [tup01[1], tup12[1], tup13[1], tup23[1]]
                out_cache = []
                for m in range(4):
                    if save_results:
                        flow_output = flow_model(inputs[m])
                    else:
                        flow_output, pr1, pr2 = flow_model(inputs[m])
                        '''Used for visualization, uncomment it needed'''
                        # flow_output1 = flow_output.cpu().detach().numpy()
                        # flow_output2 = labels[m].cpu().detach().numpy()
                        # import matplotlib.pyplot as plt
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(flow_output1[0, 0, :, :])
                        # plt.colorbar()
                        # plt.title("Output")
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(flow_output2[0, 0, :, :])
                        # plt.colorbar()
                        # plt.title("Label")
                        # plt.show()
                        flow_model.zero_grad()
                        loss_flow = SubpixelCorrNet.loss_flow([flow_output, pr1, pr2], labels[m], weights)

                        loss_flow[3].backward()
                        optimizer_flow.step()

                        flow_loss_rec.append([loss.item() for loss in loss_flow])
                    if save_results:
                        out_cache.append(F.interpolate(flow_output, (2048, 2048), mode='bilinear', align_corners=False)[0, :, :, :].cpu().detach().numpy())

                if save_results:# and not os.path.exists(train_cache_dir + str(int(idx.item())) + "_cache_flow.npy"):
                    np.save(train_cache_dir + str(int(idx.item())) + "_cache_flow.npy", np.concatenate(out_cache, axis=0))

            if epoch % step == 0 or epoch == epoch_n - 1:
                train_loss_mean = np.mean(np.array(flow_loss_rec), axis=0)

                '''Evaluation model every eval step "'''
                valid_flow_loss_rec = []
                for i, (idx, tup01, tup12, tup13, tup23) in enumerate(valid_dl_flow):
                    tup01[0], tup12[0], tup13[0], tup23[0] = tup01[0].cuda(0), tup12[0].cuda(0), tup13[0].cuda(0), tup23[0].cuda(0)
                    tup01[1], tup12[1], tup13[1], tup23[1] = tup01[1].cuda(0), tup12[1].cuda(0), tup13[1].cuda(0), tup23[1].cuda(0)
                    inputs = [tup01[0], tup12[0], tup13[0], tup23[0]]
                    labels = [tup01[1], tup12[1], tup13[1], tup23[1]]

                    out_cache = []
                    for m in range(4):
                        if save_results:
                            flow_output = flow_model(inputs[m])
                        else:
                            flow_output, pr1, pr2 = flow_model(inputs[m])
                            loss_flow = SubpixelCorrNet.loss_flow([flow_output, pr1, pr2], labels[m], weights)
                            valid_flow_loss_rec.append([loss.item() for loss in loss_flow])

                        if save_results:
                            out_cache.append(F.interpolate(flow_output, (2048, 2048), mode='bilinear', align_corners=False)[0, :, :, :].cpu().detach().numpy())

                    if save_results:
                        np.save(test_cache_dir + str(int(idx.item())) + "_cache_flow.npy",
                                np.concatenate(out_cache, axis=0))
                valid_loss_mean = np.mean(np.array(valid_flow_loss_rec), axis=0)


        elif option == "disp":
            '''Training SubpixelCorrNet with Disparity is NOT recommended, for evaluation mode to save disparity only'''
            for i, (idx, tup01, tup12, tup13, tup23) in enumerate(train_dl_disp):
                inputs = [tup01[0], tup12[0], tup13[0], tup23[0]]
                labels = [tup01[1], tup12[1], tup13[1], tup23[1]]
                out_cache_disp = []
                for m in range(4):
                    if save_results:
                        disp_output = flow_model(inputs[m])
                    else:
                        disp_output, pr1, pr2 = flow_model(inputs[m])
                        flow_model.zero_grad()
                        loss_disp = SubpixelCorrNet.loss_disp([disp_output, pr1, pr2], labels[m], weights)

                        loss_disp[3].backward()
                        optimizer_flow.step()

                        disp_loss_rec.append([loss.item() for loss in loss_disp])
                    if save_results:
                        out_cache_disp.append(F.interpolate(disp_output, (2048, 2048), mode='bilinear', align_corners=False)[0, 0, :, :].unsqueeze(0).cpu().detach().numpy())

                if save_results:
                    np.save(train_cache_dir + str(int(idx.item())) + "_cache_disp.npy", np.concatenate(out_cache_disp, axis=0))

            if epoch % step == 0 or epoch == epoch_n - 1:
                train_loss_mean = np.mean(np.array(disp_loss_rec), axis=0)
                '''Evaluation'''
                valid_disp_loss_rec = []
                for i, (idx, tup01, tup12, tup13, tup23) in enumerate(valid_dl_disp):
                    tup01[0], tup12[0], tup13[0], tup23[0] = tup01[0].cuda(0), tup12[0].cuda(0), tup13[0].cuda(0), tup23[
                        0].cuda(0)
                    tup01[1], tup12[1], tup13[1], tup23[1] = tup01[1].cuda(0), tup12[1].cuda(0), tup13[1].cuda(0), tup23[
                        1].cuda(0)
                    inputs = [tup01[0], tup12[0], tup13[0], tup23[0]]
                    labels = [tup01[1], tup12[1], tup13[1], tup23[1]]

                    out_cache = []
                    for m in range(4):
                        if save_results:
                            disp_output = flow_model(inputs[m])
                        else:
                            disp_output, pr1, pr2 = flow_model(inputs[m])
                            loss_disp = SubpixelCorrNet.loss_disp([disp_output, pr1, pr2], labels[m], weights)
                            valid_disp_loss_rec.append([loss.item() for loss in loss_disp])

                        if save_results:
                            out_cache.append(F.interpolate(disp_output, (2048, 2048), mode='bilinear', align_corners=False)[0, 0, :, :].unsqueeze(0).cpu().detach().numpy())

                    if save_results:
                        np.save(test_cache_dir + str(int(idx.item())) + "_cache_disp.npy",
                                np.concatenate(out_cache, axis=0))
                valid_loss_mean = np.mean(np.array(valid_disp_loss_rec), axis=0)

        '''Summerize'''
        if epoch % step == 0 or epoch == epoch_n - 1:

            print("Epoch %d," % (epoch),
                  " train flow loss: ", train_loss_mean[:],
                  ", valid flow loss: ", valid_loss_mean[:],
                  ", timeconsume %f" % (time.perf_counter() - now))
            now = time.perf_counter()
            flow_Recorder.append([epoch, flow_lr] + list(train_loss_mean[:]) + list(valid_loss_mean[:]))
            if (valid_loss_mean[-1]< minimum_flow_loss):
                minimum_flow_loss = valid_loss_mean[-1]
                torch.save(flow_model.state_dict(), outputdir + "/"+option+"_best.pth")
            torch.save(flow_model.state_dict(), outputdir + "/"+option+"_last.pth")
            np.savetxt(outputdir + "/"+option+"_Recorder.txt", np.array(flow_Recorder), delimiter=",")


if __name__ == '__main__':
    train()