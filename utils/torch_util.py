import shutil
import torch
from torch.autograd import Variable
from os import makedirs, remove
from os.path import exists, join, basename, dirname
import collections

def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def collate_custom(batch):
    """ Custom collate function for the Dataset class
     * It doesn't convert numpy arrays to stacked-tensors, but rather combines them in a list
     * This is useful for processing annotations of different sizes
    """    
    # this case will occur in first pass, and will convert a
    # list of dictionaries (returned by the threads by sampling dataset[idx])
    # to a unified dictionary of collated values
    if isinstance(batch[0], collections.Mapping):
        return {key: collate_custom([d[key] for d in batch]) for key in batch[0]}
    # these cases will occur in recursion
    elif torch.is_tensor(batch[0]): # for tensors, use standrard collating function
        return default_collate(batch)
    else: # for other types (i.e. lists), return as is
        return batch

class BatchTensorToVars(object):
    """Convert tensors in dict batch to vars
    """
    def __init__(self, use_cuda=True):
        self.use_cuda=use_cuda
        
    def __call__(self, batch):
        batch_var = {}
        for key,value in batch.items():
            if isinstance(value,torch.Tensor) and not self.use_cuda:
                batch_var[key] = Variable(value,requires_grad=False)
            elif isinstance(value,torch.Tensor) and self.use_cuda:
                batch_var[key] = Variable(value,requires_grad=False).cuda()
            else:
                batch_var[key] = value            
        return batch_var
    
def Softmax1D(x,dim):
    x_k = torch.max(x,dim)[0].unsqueeze(dim)
    x -= x_k.expand_as(x)
    exp_x = torch.exp(x)
    return torch.div(exp_x,torch.sum(exp_x,dim).unsqueeze(dim).expand_as(x))
    
def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir!='' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir,'best_' + model_fn))
        
def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def expand_dim(tensor,dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))
        