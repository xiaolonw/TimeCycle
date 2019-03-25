from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import random

from utils.imutils2 import *
from utils.transforms import *
import torchvision.transforms as transforms

import scipy.io as sio
import scipy.misc


# get the video frames
# two patches in the future frame, one is center, the other is one of the 8 patches around

class DavisSet(data.Dataset):
    def __init__(self, params, is_train=True):

        self.filelist = params['filelist']
        self.batchSize = params['batchSize']
        self.imgSize = params['imgSize']
        self.cropSize = params['cropSize']
        self.cropSize2 = params['cropSize2']
        self.videoLen = params['videoLen']
        self.predFrames = params['predFrames'] # 4
        self.sideEdge = params['sideEdge'] # 64


        # prediction distance, how many frames far away
        self.offset = params['offset']
        # gridSize = 3
        # self.gridSize = params['gridSize']

        self.is_train = is_train

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.lblfiles = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]
            lblfile = rows[1]

            self.jpgfiles.append(jpgfile)
            self.lblfiles.append(lblfile)

        f.close()

    def cropimg(self, img, offset_x, offset_y, cropsize):

        img = im_to_numpy(img)
        cropim = np.zeros([cropsize, cropsize, 3])
        cropim[:, :, :] = img[offset_y: offset_y + cropsize, offset_x: offset_x + cropsize, :]
        cropim = im_to_torch(cropim)

        return cropim


    def __getitem__(self, index):

        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]

        imgs = []
        lbls = []
        patches = []
        target_imgs = []

        frame_num = len(os.listdir(folder_path)) + self.videoLen

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        for i in range(frame_num):
            if i < self.videoLen:
                img_path = folder_path + "{:05d}.jpg".format(0)
                lbl_path = label_path + "{:05d}.png".format(0)
            else:
                img_path = folder_path + "{:05d}.jpg".format(i - self.videoLen)
                lbl_path = label_path + "{:05d}.png".format(i - self.videoLen)

            img = load_image(img_path)  # CxHxW
            ht, wd = img.size(1), img.size(2)
            newh, neww = ht, wd

            if ht <= wd:
                ratio  = 1.0 #float(wd) / float(ht)
                # width, height
                img = resize(img, int(self.imgSize * ratio), self.imgSize)
                newh = self.imgSize
                neww = int(self.imgSize * ratio)
            else:
                ratio  = 1.0 #float(ht) / float(wd)
                # width, height
                img = resize(img, self.imgSize, int(self.imgSize * ratio))
                newh = int(self.imgSize * ratio)
                neww = self.imgSize

            if i == 0:
                imgs = torch.Tensor(frame_num, 3, newh, neww)

            img = color_normalize(img, mean, std)
            imgs[i] = img
            lblimg  = scipy.misc.imread(lbl_path)
            lblimg  = scipy.misc.imresize( lblimg, (newh, neww), 'nearest' )

            lbls.append(lblimg.copy())

        gridx = 0
        gridy = 0

        for i in range(frame_num):

            img = imgs[i]
            ht, wd = img.size(1), img.size(2)
            newh, neww = ht, wd

            sideEdge = self.sideEdge

            gridy = int(newh / sideEdge)
            gridx = int(neww / sideEdge)

            # img = im_to_numpy(img)
            # target_imgs.append(img)

            for yid in range(gridy):
                for xid in range(gridx):

                    patch_img = img[:, yid * sideEdge: yid * sideEdge + sideEdge, xid * sideEdge: xid * sideEdge + sideEdge].clone()

                    patches.append(patch_img)


        countPatches = frame_num * gridy * gridx
        patchTensor = torch.Tensor(countPatches, 3, self.cropSize2, self.cropSize2)

        for i in range(countPatches):
            patchTensor[i, :, :, :] = patches[i]

        # for i in range(len(imgs)):
        #     imgs[i] = color_normalize(imgs[i], mean, std)


        patchTensor = patchTensor.view(frame_num, gridy * gridx, 3, self.cropSize2, self.cropSize2)

        # Meta info
        meta = {'folder_path': folder_path, 'gridx': gridx, 'gridy': gridy}

        lbls_tensor = torch.Tensor(len(lbls), newh, neww, 3)
        for i in range(len(lbls)):
            lbls_tensor[i] = torch.from_numpy(lbls[i])


        return imgs, patchTensor, lbls_tensor, meta

    def __len__(self):
        return len(self.jpgfiles)
