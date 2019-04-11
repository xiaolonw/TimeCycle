'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import cv2
import imageio

import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import models.videos.model_test as video3d

from utils import Logger, AverageMeter, mkdir_p, savefig
import models.dataset.davis_test as davis

from geotnf.transformation import GeometricTnf

from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import torch.nn.functional as F
from torch.autograd import Variable


params = {}
params['filelist'] = '/nfs.yoda/xiaolonw/davis/DAVIS/vallist.txt'
# params['batchSize'] = 24
params['imgSize'] = 320
params['cropSize'] = 320
params['cropSize2'] = 80
params['videoLen'] = 8
params['offset'] = 0
params['sideEdge'] = 80
params['predFrames'] = 1



def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options

parser.add_argument('--lr', '--learning-rate', default=2e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='/scratch/xiaolonw/pytorch_checkpoints/unsup3dnl_single_contrast', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seperate2d', type=int, default=0, help='manual seed')
parser.add_argument('--batchSize', default=1, type=int,
                    help='batchSize')
parser.add_argument('--T', default=1.0, type=float,
                    help='temperature')
parser.add_argument('--gridSize', default=9, type=int,
                    help='temperature')
parser.add_argument('--classNum', default=49, type=int,
                    help='temperature')
parser.add_argument('--lamda', default=0.1, type=float,
                    help='temperature')

parser.add_argument('--pretrained_imagenet', type=str_to_bool, nargs='?', const=True, default=False,
                    help='pretrained_imagenet')
parser.add_argument('--topk_vis', default=20, type=int,
                    help='topk_vis')

parser.add_argument('--videoLen', default=8, type=int,
                    help='predict how many frames away')
parser.add_argument('--frame_gap', default=2, type=int,
                    help='predict how many frames away')

parser.add_argument('--cropSize', default=320, type=int,
                    help='predict how many frames away')
parser.add_argument('--cropSize2', default=80, type=int,
                    help='predict how many frames away')
parser.add_argument('--temporal_out', default=4, type=int,
                    help='predict how many frames away')

parser.add_argument('--save_path', default='', type=str)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


params['batchSize'] = state['batchSize']
print('batchSize: ' + str(params['batchSize']) )

print('temperature: ' + str(state['T']))

params['gridSize'] = state['gridSize']
print('gridSize: ' + str(params['gridSize']) )

params['classNum'] = state['classNum']
print('classNum: ' + str(params['classNum']) )

params['videoLen'] = state['videoLen']
print('videoLen: ' + str(params['videoLen']) )

params['cropSize'] = state['cropSize']
print('cropSize: ' + str(params['cropSize']) )
params['imgSize'] = state['cropSize']


params['cropSize2'] = state['cropSize2']
print('cropSize2: ' + str(params['cropSize2']) )
params['sideEdge'] = state['cropSize2']


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

print(args.gpu_id)

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_loss = 0  # best test accuracy

def partial_load(pretrained_dict, model):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

def main():
    global best_loss
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    val_loader = torch.utils.data.DataLoader(
        davis.DavisSet(params, is_train=False),
        batch_size=int(params['batchSize']), shuffle=False,
        num_workers=args.workers, pin_memory=True)


    model = video3d.CycleTime(class_num=params['classNum'], trans_param_num=3, pretrained=args.pretrained_imagenet, temporal_out=args.temporal_out)
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    title = 'videonet'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        partial_load(checkpoint['state_dict'], model)

        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Contrast Loss'])

        del checkpoint

    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Contrast Loss'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss = test(val_loader, model, 1, use_cuda)




def test(val_loader, model, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    save_objs = args.evaluate

    import os
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)# /scratch/xiaolonw/davis_results_mask_mixfcn/')
    # save_path = '/scratch/xiaolonw/davis_results_mask_mixfcn/'
    save_path = args.save_path + '/'
    # img_path  = '/scratch/xiaolonw/vlog_frames/'
    save_file = '%s/list.txt' % save_path

    fileout = open(save_file, 'w')

    end = time.time()

    # bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (imgs_total, patch2_total, lbls, meta) in enumerate(val_loader):

        finput_num_ori = params['videoLen']
        finput_num     = finput_num_ori

        # measure data loading time
        data_time.update(time.time() - end)
        imgs_total = torch.autograd.Variable(imgs_total.cuda())
        # patch2_total = torch.autograd.Variable(patch2_total.cuda())

        t00 = time.time()

        bs = imgs_total.size(0)
        total_frame_num = imgs_total.size(1)
        channel_num = imgs_total.size(2)
        height_len  = imgs_total.size(3)
        width_len   = imgs_total.size(4)

        assert(bs == 1)

        folder_paths = meta['folder_path']
        gridx = int(meta['gridx'].data.cpu().numpy()[0])
        gridy = int(meta['gridy'].data.cpu().numpy()[0])
        print('gridx: ' + str(gridx) + ' gridy: ' + str(gridy))
        print('total_frame_num: ' + str(total_frame_num))

        height_dim = int(params['cropSize'] / 8)
        width_dim  = int(params['cropSize'] / 8)

        # processing labels
        lbls = lbls[0].data.cpu().numpy()
        print(lbls.shape)
        # print(patch2_total.size())

        lbls_new = []

        lbl_set = []
        lbl_set.append(np.zeros(3).astype(np.uint8))
        count_lbls = []
        count_lbls.append(0)

        for i in range(lbls.shape[0]):
            nowlbl = lbls[i].copy()
            if i == 0:
                for j in range(nowlbl.shape[0]):
                    for k in range(nowlbl.shape[1]):

                        pixellbl = nowlbl[j, k, :].astype(np.uint8)

                        flag = 0
                        for t in range(len(lbl_set)):
                            if lbl_set[t][0] == pixellbl[0] and lbl_set[t][1] == pixellbl[1] and lbl_set[t][2] == pixellbl[2]:
                                flag = 1
                                count_lbls[t] = count_lbls[t] + 1
                                break

                        if flag == 0:
                            lbl_set.append(pixellbl)
                            count_lbls.append(0)

            lbls_new.append(nowlbl)

        lbl_set_temp = []
        for i in range(len(lbl_set)):
            if count_lbls[i] > 10:
                lbl_set_temp.append(lbl_set[i])

        lbl_set = lbl_set_temp
        print(lbl_set)
        print(count_lbls)

        t01 = time.time()

        lbls_resize = np.zeros((lbls.shape[0], lbls.shape[1], lbls.shape[2], len(lbl_set)))
        lbls_resize2 = np.zeros((lbls.shape[0], height_dim, width_dim, len(lbl_set)))

        for i in range(lbls.shape[0]):
            nowlbl = lbls[i].copy()
            for j in range(nowlbl.shape[0]):
                for k in range(nowlbl.shape[1]):

                    pixellbl = nowlbl[j, k, :].astype(np.uint8)
                    for t in range(len(lbl_set)):
                        if lbl_set[t][0] == pixellbl[0] and lbl_set[t][1] == pixellbl[1] and lbl_set[t][2] == pixellbl[2]:
                            lbls_resize[i, j, k, t] = 1

        for i in range(lbls.shape[0]):
            lbls_resize2[i] = cv2.resize(lbls_resize[i], (height_dim, width_dim))


        t02 = time.time()
        print(t02 - t01, 'relabel', t01-t00, 'label')

        # print the images

        imgs_set = imgs_total.data
        imgs_set = imgs_set.cpu().numpy()
        imgs_set = imgs_set[0]
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        imgs_toprint = []

        # ref image
        for t in range(imgs_set.shape[0]):
            img_now = imgs_set[t]

            for c in range(3):
                img_now[c] = img_now[c] * std[c]
                img_now[c] = img_now[c] + mean[c]

            img_now = img_now * 255
            img_now = np.transpose(img_now, (1, 2, 0))
            img_now = cv2.resize(img_now, (img_now.shape[0] * 2, img_now.shape[1] * 2) )

            imgs_toprint.append(img_now)

            imname  = save_path + str(batch_idx) + '_' + str(t) + '_frame.jpg'
            scipy.misc.imsave(imname, img_now)

        for t in range(finput_num_ori):

            nowlbl = lbls_new[t]
            imname  = save_path + str(batch_idx) + '_' + str(t) + '_label.jpg'
            scipy.misc.imsave(imname, nowlbl)


        now_batch_size = 4

        imgs_stack = []
        patch2_stack = []

        im_num = total_frame_num - finput_num_ori

        trans_out_2_set = []
        corrfeat2_set = []

        imgs_tensor = torch.Tensor(now_batch_size, finput_num, 3, params['cropSize'], params['cropSize'])
        target_tensor = torch.Tensor(now_batch_size, 1, 3, params['cropSize'], params['cropSize'])

        imgs_tensor = torch.autograd.Variable(imgs_tensor.cuda())
        target_tensor = torch.autograd.Variable(target_tensor.cuda())


        t03 = time.time()

        for iter in range(0, im_num, now_batch_size):

            print(iter)

            startid = iter
            endid   = iter + now_batch_size

            if endid > im_num:
                endid = im_num

            now_batch_size2 = endid - startid

            for i in range(now_batch_size2):

                imgs = imgs_total[:, iter + i + 1: iter + i + finput_num_ori, :, :, :]
                imgs2 = imgs_total[:, 0, :, :, :].unsqueeze(1)
                imgs = torch.cat((imgs2, imgs), dim=1)

                imgs_tensor[i] = imgs
                target_tensor[i, 0] = imgs_total[0, iter + i + finput_num_ori]

            corrfeat2_now = model(imgs_tensor, target_tensor)
            corrfeat2_now = corrfeat2_now.view(now_batch_size, finput_num_ori, corrfeat2_now.size(1), corrfeat2_now.size(2), corrfeat2_now.size(3))

            for i in range(now_batch_size2):
                corrfeat2_set.append(corrfeat2_now[i].data.cpu().numpy())

        t04 = time.time()
        print(t04-t03, 'model forward', t03-t02, 'image prep')

        for iter in range(total_frame_num - finput_num_ori):

            if iter % 10 == 0:
                print(iter)

            imgs = imgs_total[:, iter + 1: iter + finput_num_ori, :, :, :]
            imgs2 = imgs_total[:, 0, :, :, :].unsqueeze(1)
            imgs = torch.cat((imgs2, imgs), dim=1)

            # trans_out_2, corrfeat2 = model(imgs, patch2)
            corrfeat2   = corrfeat2_set[iter]
            corrfeat2   = torch.from_numpy(corrfeat2)


            out_frame_num = int(finput_num)
            height_dim = corrfeat2.size(2)
            width_dim = corrfeat2.size(3)

            corrfeat2 = corrfeat2.view(corrfeat2.size(0), height_dim, width_dim, height_dim, width_dim)
            corrfeat2 = corrfeat2.data.cpu().numpy()


            topk_vis = args.topk_vis
            vis_ids_h = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)
            vis_ids_w = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)

            t05 = time.time()

            atten1d  = corrfeat2.reshape(corrfeat2.shape[0], height_dim * width_dim, height_dim, width_dim)
            ids = np.argpartition(atten1d, -topk_vis, axis=1)[:, -topk_vis:]
            # ids = np.argsort(atten1d, axis=1)[:, -topk_vis:]

            hid = ids // width_dim
            wid = ids % width_dim

            vis_ids_h = wid.transpose(0, 2, 3, 1)
            vis_ids_w = hid.transpose(0, 2, 3, 1)

            t06 = time.time()

            img_now = imgs_toprint[iter + finput_num_ori]

            predlbls = np.zeros((height_dim, width_dim, len(lbl_set)))
            # predlbls2 = np.zeros((height_dim * width_dim, len(lbl_set)))

            for t in range(finput_num):

                tt1 = time.time()

                h, w, k = np.meshgrid(np.arange(height_dim), np.arange(width_dim), np.arange(topk_vis), indexing='ij')
                h, w = h.flatten(), w.flatten()

                hh, ww = vis_ids_h[t].flatten(), vis_ids_w[t].flatten()

                if t == 0:
                    lbl = lbls_resize2[0, hh, ww, :]
                else:
                    lbl = lbls_resize2[t + iter, hh, ww, :]

                np.add.at(predlbls, (h, w), lbl * corrfeat2[t, ww, hh, h, w][:, None])

            t07 = time.time()
            # print(t07-t06, 'lbl proc', t06-t05, 'argsorts')

            predlbls = predlbls / finput_num

            for t in range(len(lbl_set)):
                nowt = t
                predlbls[:, :, nowt] = predlbls[:, :, nowt] - predlbls[:, :, nowt].min()
                predlbls[:, :, nowt] = predlbls[:, :, nowt] / predlbls[:, :, nowt].max()


            lbls_resize2[iter + finput_num_ori] = predlbls

            predlbls_cp = predlbls.copy()
            predlbls_cp = cv2.resize(predlbls_cp, (params['imgSize'], params['imgSize']))
            predlbls_val = np.zeros((params['imgSize'], params['imgSize'], 3))

            ids = np.argmax(predlbls_cp[:, :, 1 : len(lbl_set)], 2)

            predlbls_val = np.array(lbl_set)[np.argmax(predlbls_cp, axis=-1)]
            predlbls_val = predlbls_val.astype(np.uint8)
            predlbls_val2 = cv2.resize(predlbls_val, (img_now.shape[0], img_now.shape[1]), interpolation=cv2.INTER_NEAREST)

            # activation_heatmap = cv2.applyColorMap(predlbls, cv2.COLORMAP_JET)
            img_with_heatmap =  np.float32(img_now) * 0.5 + np.float32(predlbls_val2) * 0.5

            imname  = save_path + str(batch_idx) + '_' + str(iter + finput_num_ori) + '_label.jpg'
            imname2  = save_path + str(batch_idx) + '_' + str(iter + finput_num_ori) + '_mask.png'

            scipy.misc.imsave(imname, np.uint8(img_with_heatmap))
            scipy.misc.imsave(imname2, np.uint8(predlbls_val))



    fileout.close()

    return losses.avg


if __name__ == '__main__':
    main()
