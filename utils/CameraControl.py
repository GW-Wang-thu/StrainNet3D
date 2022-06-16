import sys
import os
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap
# sys.path.append()
from ExtImports.MvImport.MvCameraControl_class import *
# from CamOperation_class import *
import cv2
import threading
import numpy as np
import time
import inspect
import ctypes
import random
from PIL import Image, ImageTk
from ctypes import *

from PyQt5.QtCore import QTimer, QCoreApplication, Qt
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageQt

from datetime import datetime

class Your_Camera_Class():
    def __init__(self, idx):
        self.TEMP_IMG = None
        pass


def stop_thread(thread, exctype=SystemExit):
    tid = thread.ident
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class Camera:
    '''Camera Demo for images acquisition'''
    def __init__(self, idx, name):
        self.devicelist = ["Camera1", "Camera2"]
        self.idx = idx
        self.name = name
        self.is_open = False
        self.is_grabbing = False
        self.is_threading = False
        self.is_to_saveframe = False
        self.cam = Your_Camera_Class(idx)

        self.crop11 = 0
        self.crop12 = -1
        self.crop21 = 0
        self.crop22 = -1
        self.temp_img = None

    def open_device(self):
        ''' Start Gribbing code'''
        try:
            self.n_win_gui_id = random.randint(1, 10000)
            self.h_thread_handle = threading.Thread(target=Camera.Work_thread)
            self.h_thread_handle.start()
            self.is_threading = True
        except:
            self.is_grabbing = False
        pass
        return 0

    def stop_grabbing(self):
        '''Stop threading and close device'''
        stop_thread(thread=self.h_thread_handle)
        self.is_threading = False
        self.is_grabbing = False
        self.is_open = False
        pass
        return 0

    def Work_thread(self):
        '''Grib Work Thread '''
        while True:
            Img = self.cam.TEMP_IMG
            self.temp_img = Img
