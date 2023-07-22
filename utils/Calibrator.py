import os.path
import time
import cv2
import numpy as np


class Stereo_calibrator():
    def __init__(self, num_row, num_cow, physicalsize, tmp_img):
        self.criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 10000, 1e-12)
        self.stereo_criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 10000, 1e-12)
        self.num_row = num_row
        self.num_col = num_cow
        self.ruler = physicalsize
        self.tmp_img = tmp_img
        self.limg_points = []
        self.rimg_points = []
        self.obj_points = []
        self.obj_point_tmp = np.zeros((self.num_row * self.num_col, 3), np.float32)
        self.obj_point_tmp[:, :2] = np.mgrid[0:self.num_row, 0:self.num_col].T.reshape(-1, 2) * self.ruler
        self.l_marked_img = []
        self.r_marked_img = []

    def find_corners(self, img_l, img_r):
        now = time.perf_counter()
        ret_l, corners_l = cv2.findChessboardCorners(img_l, (self.num_row, self.num_col), self.criteria)
        ret_r, corners_r = cv2.findChessboardCorners(img_r, (self.num_row, self.num_col), self.criteria)
        print(ret_l, ret_r)
        temp_cornerd_img_l=None
        temp_cornerd_img_r=None
        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(img_l, corners_l, (9, 9), (-1, -1), self.criteria)
            corners_r = cv2.cornerSubPix(img_r, corners_r, (9, 9), (-1, -1), self.criteria)
            print(time.perf_counter() - now)
            self.limg_points.append(corners_l)
            self.rimg_points.append(corners_r)
            self.obj_points.append(self.obj_point_tmp)
            temp_cornerd_img_l = cv2.drawChessboardCorners(cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR), (self.num_row, self.num_col), corners_l, ret_l)
            temp_cornerd_img_r = cv2.drawChessboardCorners(cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR), (self.num_row, self.num_col), corners_r, ret_r)
            self.l_marked_img.append(temp_cornerd_img_l)
            self.r_marked_img.append(temp_cornerd_img_r)
            cv2.imshow("L", cv2.resize(temp_cornerd_img_l, None, fx=0.4, fy=0.4))
            cv2.imshow("R", cv2.resize(temp_cornerd_img_r, None, fx=0.4, fy=0.4))
            cv2.waitKey()

        return ret_l and ret_r

    def calibrate(self,
                  CALIB_FIX_ASPECT_RATIO=True,
                  CALIB_FIX_PRINCIPAL_POINT = True,
                  CALIB_USE_INTRINSIC_GUESS=True,
                  CALIB_SAME_FOCAL_LENGTH=False,
                  CALIB_ZERO_TANGENT_DIST=False,
                  CALIB_RATIONAL_MODEL=False,
                  No_Distortion=True):
        flags = 0
        if CALIB_FIX_PRINCIPAL_POINT:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        if CALIB_FIX_ASPECT_RATIO:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        if CALIB_USE_INTRINSIC_GUESS:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        if CALIB_SAME_FOCAL_LENGTH:
            flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        if CALIB_ZERO_TANGENT_DIST:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
        if CALIB_RATIONAL_MODEL:
            flags |= cv2.CALIB_RATIONAL_MODEL
        if No_Distortion:
            flags |= cv2.CALIB_FIX_K1
            flags |= cv2.CALIB_FIX_K2
            flags |= cv2.CALIB_FIX_K3
            flags |= cv2.CALIB_FIX_K4
            flags |= cv2.CALIB_FIX_K5
            flags |= cv2.CALIB_FIX_K6
        ret1, m_l, d_l, rvects_l, tvects_l = cv2.calibrateCamera(self.obj_points, self.limg_points, self.tmp_img.shape[::-1], None, None)
        ret2, m_r, d_r, rvects_r, tvects_r = cv2.calibrateCamera(self.obj_points, self.rimg_points, self.tmp_img.shape[::-1], None, None)
        ret3, m1, d1, m2, d2, R, t, E, F = cv2.stereoCalibrate(self.obj_points,
                                                              self.limg_points,
                                                              self.rimg_points,
                                                              m_l,
                                                              d_l,
                                                              m_r,
                                                              d_r,
                                                              self.tmp_img.shape[::-1],
                                                              criteria=self.stereo_criteria,
                                                              flags=flags)
        # 构建单应性矩阵
        print(ret1, '\n左相机矩阵：%s\n左相机畸变:%s\n右相机矩阵：%s\n右相机畸变:%s\n旋转矩阵:%s\n平移向量:%s'%(m1, d1, m2, d2, R, t))
        error = {}
        mean_error_l = 0
        mean_error_r = 0

        for i in range(len(self.obj_points)):
            l_image_position, _ = cv2.projectPoints(self.obj_points[i], rvects_l[i], tvects_l[i], m_l, d_l)
            l_error = cv2.norm(self.limg_points[i], l_image_position, cv2.NORM_L2) / len(l_image_position)


            r_image_position, _ = cv2.projectPoints(self.obj_points[i], rvects_r[i], tvects_r[i], m_r, d_r)
            r_error = cv2.norm(self.rimg_points[i], r_image_position, cv2.NORM_L2) / len(r_image_position)

            cv2.imshow("L", cv2.resize(
                cv2.drawChessboardCorners(self.l_marked_img[i], (self.num_row, self.num_col), l_image_position, 1),
                None, fx=0.4, fy=0.4))
            cv2.imshow("R", cv2.resize(
                cv2.drawChessboardCorners(self.r_marked_img[i], (self.num_row, self.num_col), r_image_position, 1),
                None, fx=0.4, fy=0.4))
            cv2.waitKey()

            error.update({'errror_l_idx'+str(i): round(l_error, 5),
                          'errror_r_idx'+str(i): round(r_error, 5)})
            mean_error_l += l_error
            mean_error_r += r_error

        error.update({"errror_l_Avg":mean_error_l / len(self.obj_points),
                      "errror_r_Avg":mean_error_r / len(self.obj_points)})

        if ret1 and ret2 and ret3:
            return (m1, d1, m2, d2, R, t), error
        else:
            return None, None

if __name__ == '__main__':
    dir = r'D:\Codes\HICCS_Series\HICCS_FF\Workdir\test20220326\Calibimg\\'
    my_calibrator = Stereo_calibrator(11, 8, 6.0, tmp_img=cv2.imread(dir + '0_0.bmp', cv2.IMREAD_GRAYSCALE))
    i = 0

    for i in range(3):
        if os.path.exists(dir + str(i) + "_0.bmp"):
            ret = my_calibrator.find_corners(img_l=cv2.imread(dir + str(i) + "_0.bmp", cv2.IMREAD_GRAYSCALE),
                                             img_r=cv2.imread(dir + str(i) + "_1.bmp", cv2.IMREAD_GRAYSCALE))
            print(ret)

    ret, error = my_calibrator.calibrate()
    if ret is not None:
        print(ret, '\n左相机矩阵：%s\n左相机畸变:%s\n右相机矩阵：%s\n右相机畸变:%s\n旋转矩阵:%s\n平移向量:%s' % ret, error)
