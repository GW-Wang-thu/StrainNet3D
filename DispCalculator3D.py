import time

import numpy as np
import torch
from Models import SubpixelCorrNet, FusionNet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import json


class DispCalculator:

    def __init__(self):
        self.cropbox = [0, -1, 0, -1]
        self.resolution = (3000, 4096)
        self.LRRR_img_torch = None
        self.lr = None
        self.lr_full = None
        self.rr_full = None
        self.Disparity0 = None
        self.inputTrans = transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))

    def load_model(self,
                   params_file="../Consts/Calib.json",
                   disp_model_params="../Consts/dispnet_param.pth",
                   flow_model_params="Consts/flownet_param.pth",
                   fuse_model_params="Consts/flownet_param.pth",
                   proj_disparity_y_file=None,
                   crop_box=[500, 1500, 1000, 3000],
                   Gaussian_Filter=5,
                   cudaid=0):

        with open(params_file, "r") as f:
            self.params = json.load(f)
        self.Tx_r = self.params["RTX"]
        self.Tz_r = self.params["RTZ"]
        self.cx_r = self.params["RCX"]
        self.fx_r = self.params["RFX"]

        self.cx_l = self.params["LCX"]
        self.cy_l = self.params["LCY"]
        self.fx_l = self.params["LFX"]
        self.fy_l = self.params["LFY"]
        self.RRot = np.array(self.params["RRot"])
        self.flag_Got_Cameracalibrationfile = True

        self.cudaid = cudaid
        self.disp_model = SubpixelCorrNet.SubpixelCorrNet().cuda(cudaid)
        self.flow_model = SubpixelCorrNet.SubpixelCorrNet().cuda(cudaid)
        box = [10, 2038, 10, 2038]
        stp = (500, 1000)
        self.fuse_model = FusionNet.FusionNet_S(params=params_file,
                                                cropbox=[stp[0]+box[0], stp[0]+box[1], stp[1]+box[2], stp[1]+box[3]]).cuda(self.cudaid)
        self.disp_model.eval()
        self.flow_model.eval()
        self.fuse_model.load_state_dict(torch.load(fuse_model_params))
        self.fuse_model.eval()
        disp_params = torch.load(disp_model_params)
        self.disp_model.load_state_dict(disp_params)
        flow_params = torch.load(flow_model_params)
        self.flow_model.load_state_dict(flow_params)
        self.GaussianFiltersize = Gaussian_Filter

        self.cropbox = crop_box
        self.shift_LX, self.shift_LY = self._generate_l_array(self.cropbox)
        self.shift_LX = torch.from_numpy(self.shift_LX).cuda(self.cudaid)
        self.shift_LY = torch.from_numpy(self.shift_LY).cuda(self.cudaid)
        if proj_disparity_y_file is None:
            self.proj_disparity_y = torch.from_numpy(
                np.zeros(shape=(self.cropbox[1] - self.cropbox[0], self.cropbox[3] - self.cropbox[2]))).cuda(self.cudaid)
        else:
            self.proj_disparity_y = torch.from_numpy(
                np.loadtxt(proj_disparity_y_file)[self.cropbox[0]:self.cropbox[1],
                self.cropbox[2]:self.cropbox[3]]).cuda(self.cudaid)  #Donnot dismiss y disparity在投影的基础上计算，仍会存在的一小部分视差，y方向

        self.h = self.cropbox[1] - self.cropbox[0]
        self.w = self.cropbox[3] - self.cropbox[2]

        self.GaussianFiltersize = Gaussian_Filter
        return 0

    def _Get_Transpose_Matrix(self):
        lbk = np.zeros(shape=self.resolution, dtype="uint8")
        lbk[self.cropbox[0]:self.cropbox[0] + self.lr.shape[0],
        self.cropbox[2]:self.cropbox[2] + self.lr.shape[1]] = self.lr
        rbk = np.zeros(shape=self.resolution, dtype="uint8")
        rbk[self.cropbox[0]:self.cropbox[0] + self.rr.shape[0],
        self.cropbox[2]:self.cropbox[2] + self.rr.shape[1]] = self.rr

        akaze = cv2.AKAZE_create()

        kp1, des1 = akaze.detectAndCompute(rbk, None)
        kp2, des2 = akaze.detectAndCompute(lbk, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.3 * n.distance:
                good_matches.append([m])

        src_automatic_points = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        den_automatic_points = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, status = cv2.findHomography(src_automatic_points, den_automatic_points, cv2.RANSAC, 5.0)
        print("Finish Calculation of Transpose Matrix")
        return H

    def _Get_Avg_Matrix(self, img, windsize=60):        # Gray adjustment
        imsize = img.shape
        avg_matrix = []
        for i in range(imsize[0]//windsize):
            start_x = i * windsize
            t_list = []
            for j in range(imsize[1]//windsize):
                start_y = j * windsize
                t_block = img[start_x:start_x+windsize, start_y:start_y+windsize]
                avg = np.average(t_block)
                t_list.append(avg)
            avg_matrix.append(t_list)
        avg_matrix = cv2.resize(np.array(avg_matrix, dtype="uint8"), dsize=(imsize[1], imsize[0]), interpolation=cv2.INTER_CUBIC)

        return avg_matrix, np.average(avg_matrix)

    def _project_r2l(self, right_img):
        warped_automatic_image = cv2.warpPerspective(right_img, self.H, (right_img.shape[1], right_img.shape[0]),
                                                     flags=cv2.INTER_CUBIC)
        return warped_automatic_image

    def _left_to_right(self, right_projed_coord_x, right_projed_coord_y):
        ur = (self.Transpose_matrix[0, 0] * right_projed_coord_x + self.Transpose_matrix[0, 1] * right_projed_coord_y +
              self.Transpose_matrix[0, 2]) / (
                         self.Transpose_matrix[2, 0] * right_projed_coord_x + self.Transpose_matrix[
                     2, 1] * right_projed_coord_y + self.Transpose_matrix[2, 2])
        vr = (self.Transpose_matrix[1, 0] * right_projed_coord_x + self.Transpose_matrix[1, 1] * right_projed_coord_y +
              self.Transpose_matrix[1, 2]) / (
                         self.Transpose_matrix[2, 0] * right_projed_coord_x + self.Transpose_matrix[
                     2, 1] * right_projed_coord_y + self.Transpose_matrix[2, 2])
        return ur, vr


    def _solve_displacement(self, flow_x, flow_y, proj_disparity_x, proj_disparity_x_1):

        # proj_disparity_y = np.loadtxt("L:\ExpData\EXP_PAPER\DATASET\coordinates\\Plane_Disparity_RY.csv")
        right_projed_coord_x = self.shift_LX + proj_disparity_x
        right_projed_coord_y = self.shift_LY + self.proj_disparity_y
        right_coord_x, right_coord_y = self._left_to_right(right_projed_coord_x, right_projed_coord_y)
        xw_0, yw_0, zw_0 = self._cal_surface_coord_world(self.shift_LX + 1.0, self.shift_LY + 1.0, right_coord_x + 1.0)

        # After Deformation
        left_coord_x_1 = self.shift_LX + flow_x
        left_coord_y_1 = self.shift_LY + flow_y
        right_projed_coord_x_1 = left_coord_x_1 + proj_disparity_x_1
        right_projed_coord_y_1 = left_coord_y_1 + self.proj_disparity_y
        right_coord_x_1, right_coord_y_1 = self._left_to_right(right_projed_coord_x_1, right_projed_coord_y_1)
        xw_1, yw_1, zw_1 = self._cal_surface_coord_world(left_coord_x_1 + 1.0, left_coord_y_1 + 1.0,
                                                         right_coord_x_1 + 1.0)

        return (xw_0, yw_0, zw_0), (xw_1 - xw_0, yw_1 - yw_0, zw_1 - zw_0)

    def _load_shift_array(self, filename, crop):
        shift_arrays = np.load(filename)
        return shift_arrays[0, crop[0]:crop[1], crop[2]:crop[3]], \
               shift_arrays[1, crop[0]:crop[1], crop[2]:crop[3]], \
               shift_arrays[2, crop[0]:crop[1], crop[2]:crop[3]], \
               shift_arrays[3, crop[0]:crop[1], crop[2]:crop[3]]

    def _generate_l_array(self, crop):
        lx = np.expand_dims(np.arange(crop[2], crop[3], 1), 0).repeat(crop[1] - crop[0], axis=0)
        ly = np.expand_dims(np.arange(crop[0], crop[1], 1), 1).repeat(crop[3] - crop[2], axis=1)
        return lx, ly

    def _cal_surface_coord_world(self, ul, vl, ur):

        zw = (self.Tx_r - self.Tz_r * (ur - self.cx_r) / self.fx_r) / \
             (((ur - self.cx_r) / self.fx_r) * (self.RRot[2, 0] * (ul - self.cx_l) / self.fx_l + self.RRot[2, 1] * (
                         vl - self.cy_l) / self.fy_l + self.RRot[2, 2]) -
              (self.RRot[0, 0] * (ul - self.cx_l) / self.fx_l + self.RRot[0, 1] * (vl - self.cy_l) / self.fy_l +
               self.RRot[0, 2]))
        xw = zw * (ul - self.cx_l) / self.fx_l
        yw = zw * (vl - self.cy_l) / self.fy_l

        return xw, yw, zw


    def refresh_lrrr(self, lr, rr):
        self.lr_full = lr
        self.rr_full = rr
        self.lr = lr[self.cropbox[0]:self.cropbox[1], self.cropbox[2]:self.cropbox[3]]
        self.rr = rr[self.cropbox[0]:self.cropbox[1], self.cropbox[2]:self.cropbox[3]]

        '''Calculate Transpose Matrix, Only need one time for sequential processing'''
        self.H = self._Get_Transpose_Matrix()

        self.Transpose_matrix = np.linalg.inv(self.H)
        self.fuse_model.Transpose_matrix = self.H

        rr = self._project_r2l(rr.astype('uint8'))
        rr = rr[self.cropbox[0]:self.cropbox[1], self.cropbox[2]:self.cropbox[3]]

        self.avg_matrix_l, self.avg_l = self._Get_Avg_Matrix(self.lr)
        self.avg_matrix_r, self.avg_r = self._Get_Avg_Matrix(rr)

        self.lr = 0.5 * (self.avg_r + self.avg_l) * self.lr / self.avg_matrix_l
        self.lr = (self.lr - 255) * (self.lr <= 255) + 255
        self.lr = cv2.GaussianBlur(self.lr.astype('uint8'), ksize=(self.GaussianFiltersize, self.GaussianFiltersize),
                                   sigmaX=self.GaussianFiltersize // 2)

        rr = 0.5 * (self.avg_r + self.avg_l) * rr / self.avg_matrix_r
        rr = (rr - 255) * (rr <= 255) + 255
        rr = cv2.GaussianBlur(rr.astype('uint8'), ksize=(self.GaussianFiltersize, self.GaussianFiltersize), sigmaX=self.GaussianFiltersize // 2)

        LRRR_imgs_init = np.stack([self.lr, rr], axis=0).astype("float32")
        LRRR_imgs = (LRRR_imgs_init - np.min(LRRR_imgs_init)) / (np.max(LRRR_imgs_init) - np.min(LRRR_imgs_init))
        self.LRRR_img_torch = self.inputTrans(torch.from_numpy(LRRR_imgs)).unsqueeze(0).cuda()

        now = time.perf_counter()
        disparity0 = F.interpolate(self.disp_model(self.LRRR_img_torch), (self.h, self.w), mode='bicubic',
                                   align_corners=False)
        print('Disp_0', time.perf_counter()-now)
        self.disp_0 = disparity0

        right_projed_coord_x = self.shift_LX + disparity0[0, 0, :, :]
        right_projed_coord_y = self.shift_LY + disparity0[0, 1, :, :]

        right_coord_x = (self.Transpose_matrix[0, 0] * right_projed_coord_x + self.Transpose_matrix[
            0, 1] * right_projed_coord_y + self.Transpose_matrix[0, 2]) \
                        / (self.Transpose_matrix[2, 0] * right_projed_coord_x + self.Transpose_matrix[
            2, 1] * right_projed_coord_y + self.Transpose_matrix[2, 2])
        # right_coord_y = (self.Transpose_matrix[1, 0] * right_projed_coord_x + self.Transpose_matrix[1, 1] * right_projed_coord_y +self.Transpose_matrix[1, 2])\
        #                 / (self.Transpose_matrix[2, 0] * right_projed_coord_x + self.Transpose_matrix[2, 1] * right_projed_coord_y + self.Transpose_matrix[2, 2])

        zw_0 = (self.Tx_r - self.Tz_r * (right_coord_x + 1.0 - self.cx_r) / self.fx_r) / \
               (((right_coord_x + 1.0 - self.cx_r) / self.fx_r) *
                (self.RRot[2, 0] * (self.shift_LX + 1.0 - self.cx_l) / self.fx_l + self.RRot[2, 1] * (
                            self.shift_LY + 1.0 - self.cy_l) / self.fy_l + self.RRot[2, 2])
                - (self.RRot[0, 0] * (self.shift_LX + 1.0 - self.cx_l) / self.fx_l + self.RRot[0, 1] * (
                                   self.shift_LY + 1.0 - self.cy_l) / self.fy_l + self.RRot[0, 2]))
        xw_0 = zw_0 * (self.shift_LX + 1.0 - self.cx_l) / self.fx_l
        yw_0 = zw_0 * (self.shift_LY + 1.0 - self.cy_l) / self.fy_l
        self.zw_0 = zw_0
        self.xw_0 = xw_0
        self.yw_0 = yw_0

    def CalculateDisp3D(self, ld, rd, calc_refined=False):
        self.zoom_level = 1.0
        self.temp_zoom_loc = (0, 0)
        self.show_box = [0, self.cropbox[1] - self.cropbox[0], 0, self.cropbox[3] - self.cropbox[2]]
        self.temp_stp = (0, 0)
        self.stp_0 = None
        self.temp_zoom_level_all = None
        temp_zoom_loc_init = (0, 0)

        rd = self._project_r2l(rd)
        ld = ld[self.cropbox[0]:self.cropbox[1], self.cropbox[2]:self.cropbox[3]]
        rd = rd[self.cropbox[0]:self.cropbox[1], self.cropbox[2]:self.cropbox[3]]

        ld = 0.5 * (self.avg_r + self.avg_l) * ld / self.avg_matrix_l
        rd = 0.5 * (self.avg_r + self.avg_l) * rd / self.avg_matrix_r

        ld = (ld - 255) * (ld <= 255) + 255
        rd = (rd - 255) * (rd <= 255) + 255

        ld = cv2.GaussianBlur(ld.astype('uint8'), ksize=(self.GaussianFiltersize, self.GaussianFiltersize),
                              sigmaX=self.GaussianFiltersize // 2)
        rd = cv2.GaussianBlur(rd.astype('uint8'), ksize=(self.GaussianFiltersize, self.GaussianFiltersize),
                              sigmaX=self.GaussianFiltersize // 2)

        # Image to Tensor
        LRLD_imgs_init = np.stack([self.lr, ld], axis=0).astype("float32")
        LRLD_imgs = (LRLD_imgs_init - np.min(LRLD_imgs_init)) / (np.max(LRLD_imgs_init) - np.min(LRLD_imgs_init))
        LRLD_img_torch = self.inputTrans(torch.from_numpy(LRLD_imgs)).unsqueeze(0).cuda()

        LDRD_imgs_init = np.stack([ld, rd], axis=0).astype("float32")
        LDRD_imgs = (LDRD_imgs_init - np.min(LDRD_imgs_init)) / (np.max(LDRD_imgs_init) - np.min(LDRD_imgs_init))
        LDRD_img_torch = self.inputTrans(torch.from_numpy(LDRD_imgs)).unsqueeze(0).cuda()

        # Calculate Disparity and Flow
        now = time.perf_counter()
        flow_output = F.interpolate(self.flow_model(LRLD_img_torch), (self.h, self.w), mode='bicubic',
                                    align_corners=False)
        disp_output_1 = F.interpolate(self.disp_model(LDRD_img_torch), (self.h, self.w), mode='bicubic',
                                      align_corners=False)
        print('Disp_1 and Flow 12:', time.perf_counter() - now)

        # After Deformation
        left_coord_x_1 = self.shift_LX + flow_output[0, 0, :, :]
        left_coord_y_1 = self.shift_LY + flow_output[0, 1, :, :]
        right_projed_coord_x_1 = left_coord_x_1 + disp_output_1[0, 0, :, :]
        right_projed_coord_y_1 = left_coord_y_1 + disp_output_1[0, 1, :, :]
        right_coord_x_1 = (self.Transpose_matrix[0, 0] * right_projed_coord_x_1 + self.Transpose_matrix[
            0, 1] * right_projed_coord_y_1 + self.Transpose_matrix[0, 2]) / (
                                  self.Transpose_matrix[2, 0] * right_projed_coord_x_1 + self.Transpose_matrix[
                              2, 1] * right_projed_coord_y_1 + self.Transpose_matrix[2, 2])
        # right_coord_y_1 = (self.Transpose_matrix[1, 0] * right_projed_coord_x_1 + self.Transpose_matrix[1, 1] * right_projed_coord_y_1 + self.Transpose_matrix[1, 2]) / (
        #                  self.Transpose_matrix[2, 0] * right_projed_coord_x_1 + self.Transpose_matrix[2, 1] * right_projed_coord_y_1 + self.Transpose_matrix[2, 2])

        # xw_1, yw_1, zw_1 = self._cal_surface_coord_world(left_coord_x_1 + 1.0, left_coord_y_1 + 1.0, right_coord_x_1 + 1.0)
        zw_1 = (self.Tx_r - self.Tz_r * (right_coord_x_1 + 1.0 - self.cx_r) / self.fx_r) / \
               (((right_coord_x_1 + 1.0 - self.cx_r) / self.fx_r) * (
                           self.RRot[2, 0] * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l + self.RRot[2, 1] * (
                           left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l + self.RRot[2, 2]) -
                (self.RRot[0, 0] * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l + self.RRot[0, 1] * (
                            left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l +
                 self.RRot[0, 2]))
        xw_1 = zw_1 * (left_coord_x_1 + 1.0 - self.cx_l) / self.fx_l
        yw_1 = zw_1 * (left_coord_y_1 + 1.0 - self.cy_l) / self.fy_l

        u = (xw_1 - self.xw_0).cpu().detach().numpy()
        v = (yw_1 - self.yw_0).cpu().detach().numpy()
        w = (zw_1 - self.zw_0).cpu().detach().numpy()
        if calc_refined:
            _, refined = self.fuse_model(xw_1 - self.xw_0,
                                         yw_1 - self.yw_0,
                                         zw_1 - self.zw_0,
                                         key=1)
            refined = refined[0].cpu().detach().numpy()
            u_refine = refined[0, :, :]
            v_refine = refined[1, :, :]
            w_refine = refined[2, :, :]
            return u, v, w, u_refine, v_refine, w_refine

        return u, v, w


def gradient_filter(displacement, filter_size, step_length):
    u = displacement[0]
    v = displacement[1]
    w = displacement[2]
    center=[0]
    kernal = []
    for i in range(filter_size//2):
        m=i+1
        center.append(1/(2*m))
        center.insert(0, -1/(2*m))
        kernal.append(center)
        kernal.append(center)
    kernal.append(center)
    avg_num = filter_size * (filter_size-1) / 2
    kernal_x = (1/(step_length * avg_num)) * np.array(kernal)
    kernal_y = kernal_x.T
    u_x = cv2.filter2D(u, -1, kernal_x)
    v_y = cv2.filter2D(v, -1, kernal_y)
    w_x = cv2.filter2D(w, -1, kernal_x)
    w_y = cv2.filter2D(w, -1, kernal_y)

    exx = u_x + 0.5 * w_x**2
    eyy = v_y + 0.5 * w_y**2

    return exx, eyy


if __name__ == '__main__':

    params_dir = r'F:\case\NetParams\\'
    img_dir = r"I:\DLDIC_3D_Dataset\Paper_Exp\Exp\ExtremLight\\"
    my_disp_calculator = DispCalculator()

    my_disp_calculator.load_model(params_file=params_dir + "Calib.json",
                                  disp_model_params=params_dir + "flow_best.pth",
                                  flow_model_params=params_dir + "flow_best.pth",
                                  fuse_model_params=params_dir + "fuse_best.pth",
                                  proj_disparity_y_file=None,
                                  crop_box=[400, 2576, 1000, 3112],
                                  Gaussian_Filter=5,
                                  cudaid=0)

    my_disp_calculator.refresh_lrrr(lr=cv2.imread(img_dir + "FL_0_c.bmp", cv2.IMREAD_GRAYSCALE),
                                    rr=cv2.imread(img_dir + "FR_0_c.bmp", cv2.IMREAD_GRAYSCALE))

    u, v, w, u_refine, v_refine, w_refine = my_disp_calculator.CalculateDisp3D(ld=cv2.imread(img_dir + "FL_1_c.bmp", cv2.IMREAD_GRAYSCALE),
                                                                               rd=cv2.imread(img_dir + "FR_1_c.bmp", cv2.IMREAD_GRAYSCALE),
                                                                               calc_refined=True)
    # Calculate Strain
    exx, eyy = gradient_filter(displacement=[u, v, w], filter_size=7, step_length=1)
    plt.imshow(u)
    plt.colorbar()
    plt.show()
    plt.imshow(v)
    plt.colorbar()
    plt.show()
    plt.imshow(w)
    plt.colorbar()
    plt.show()
    plt.imshow(u_refine)
    plt.colorbar()
    plt.show()
    plt.imshow(v_refine)
    plt.colorbar()
    plt.show()
    plt.imshow(w_refine)
    plt.colorbar()
    plt.show()
    plt.imshow(exx)
    plt.colorbar()
    plt.show()





