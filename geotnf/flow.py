import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from geotnf.point_tnf import normalize_axis, unnormalize_axis

def read_flo_file(filename,verbose=False):
    """
    Read from .flo optical flow file (Middlebury format)
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    
    adapted from https://github.com/liruoteng/OpticalFlowToolkit/
    
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        raise TypeError('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        if verbose:
            print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d

def write_flo_file(flow, filename):
    """
    Write optical flow in Middlebury .flo format
    
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    
    from https://github.com/liruoteng/OpticalFlowToolkit/
    
    """
    # forcing conversion to float32 precision
    flow = flow.astype(np.float32)
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


def warp_image(image, flow):
    """
    Warp image (np.ndarray, shape=[h_src,w_src,3]) with flow (np.ndarray, shape=[h_tgt,w_tgt,2])
    
    """
    h_src,w_src=image.shape[0],image.shape[1]
    sampling_grid_torch = np_flow_to_th_sampling_grid(flow, h_src, w_src)
    image_torch = Variable(torch.FloatTensor(image.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(0))
    warped_image_torch = F.grid_sample(image_torch, sampling_grid_torch)
    warped_image = warped_image_torch.data.squeeze(0).transpose(0,1).transpose(1,2).numpy().astype(np.uint8)
    return warped_image

def np_flow_to_th_sampling_grid(flow,h_src,w_src,use_cuda=False):
    h_tgt,w_tgt=flow.shape[0],flow.shape[1]
    grid_x,grid_y = np.meshgrid(range(1,w_tgt+1),range(1,h_tgt+1))
    disp_x=flow[:,:,0]
    disp_y=flow[:,:,1]
    source_x=grid_x+disp_x
    source_y=grid_y+disp_y
    source_x_norm=normalize_axis(source_x,w_src) 
    source_y_norm=normalize_axis(source_y,h_src) 
    sampling_grid=np.concatenate((np.expand_dims(source_x_norm,2),
                                  np.expand_dims(source_y_norm,2)),2)
    sampling_grid_torch = Variable(torch.FloatTensor(sampling_grid).unsqueeze(0))
    if use_cuda:
        sampling_grid_torch = sampling_grid_torch.cuda()
    return sampling_grid_torch

# def th_sampling_grid_to_np_flow(source_grid,h_src,w_src):
#     batch_size = source_grid.size(0)
#     h_tgt,w_tgt=source_grid.size(1),source_grid.size(2)
#     source_x_norm=source_grid[:,:,:,0]
#     source_y_norm=source_grid[:,:,:,1]
#     source_x=unnormalize_axis(source_x_norm,w_src) 
#     source_y=unnormalize_axis(source_y_norm,h_src) 
#     source_x=source_x.data.cpu().numpy()
#     source_y=source_y.data.cpu().numpy()
#     grid_x,grid_y = np.meshgrid(range(1,w_tgt+1),range(1,h_tgt+1))
#     grid_x = np.repeat(grid_x,axis=0,repeats=batch_size)
#     grid_y = np.repeat(grid_y,axis=0,repeats=batch_size)
#     disp_x=source_x-grid_x
#     disp_y=source_y-grid_y
#     flow = np.concatenate((np.expand_dims(disp_x,3),np.expand_dims(disp_y,3)),3)
#     return flow

def th_sampling_grid_to_np_flow(source_grid,h_src,w_src):
    # remove batch dimension
    source_grid = source_grid.squeeze(0)
    # get mask
    in_bound_mask=(source_grid.data[:,:,0]>-1) & (source_grid.data[:,:,0]<1) & (source_grid.data[:,:,1]>-1) & (source_grid.data[:,:,1]<1)
    in_bound_mask=in_bound_mask.cpu().numpy()
    # convert coords
    h_tgt,w_tgt=source_grid.size(0),source_grid.size(1)
    source_x_norm=source_grid[:,:,0]
    source_y_norm=source_grid[:,:,1]
    source_x=unnormalize_axis(source_x_norm,w_src) 
    source_y=unnormalize_axis(source_y_norm,h_src) 
    source_x=source_x.data.cpu().numpy()
    source_y=source_y.data.cpu().numpy()
    grid_x,grid_y = np.meshgrid(range(1,w_tgt+1),range(1,h_tgt+1))
    disp_x=source_x-grid_x
    disp_y=source_y-grid_y
    # apply mask
    disp_x = disp_x*in_bound_mask+1e10*(1-in_bound_mask)
    disp_y = disp_y*in_bound_mask+1e10*(1-in_bound_mask)
    flow = np.concatenate((np.expand_dims(disp_x,2),np.expand_dims(disp_y,2)),2)
    return flow

