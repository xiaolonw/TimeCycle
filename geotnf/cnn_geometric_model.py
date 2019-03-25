from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import numpy.matlib
# from utils.torch_util import Softmax1D
from geotnf.transformation import GeometricTnf

def featureL2Norm(feature):
    epsilon = 1e-6
    #        print(feature.size())
    #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)

class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='vgg', normalization=True, last_layer='', use_cuda=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                         'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                         'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                         'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                         'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']
            if last_layer=='':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx+1])
        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer=='':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])

        if feature_extraction_cnn == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer=='':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])

        if feature_extraction_cnn == 'resnet50_scratch':
            print('resnet50_scratch')
            self.model = models.resnet50(pretrained=False)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer=='':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])

        if feature_extraction_cnn == 'resnet101_v2':
            self.model = models.resnet101(pretrained=True)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to denseblock3
            # self.model = nn.Sequential(*list(self.model.features.children())[:-3])
            # keep feature extraction network up to transitionlayer2
            self.model = nn.Sequential(*list(self.model.features.children())[:-4])
        if not train_fe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        # if use_cuda:
        #     self.model = self.model.cuda()

    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization:
            features = featureL2Norm(features)
        return features

class FeatureCorrelation(torch.nn.Module):
    def __init__(self,shape='3D',normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape=shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        if self.shape=='3D':
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
            feature_B = feature_B.view(b,c,h*w).transpose(1,2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B,feature_A)
            # indexed [batch,idx_A=col_A+h*row_A,row_B,col_B], because it uses feature_A.transpose(2,3)
            correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)

        elif self.shape=='4D':
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b,c,h*w).transpose(1,2) # size [b,c,h*w]
            feature_B = feature_B.view(b,c,h*w) # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A,feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h,w).unsqueeze(1)


        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor


class FeatureRegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True, batch_normalization=True, kernel_sizes=[7,5], channels=[128,64] ,feature_size=15):
        super(FeatureRegression, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                ch_in = feature_size*feature_size
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=0))
            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        self.linear = nn.Linear(ch_out * k_size * k_size, output_dim)
        # if use_cuda:
        #     self.conv.cuda()
        #     self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class CNNGeometric(nn.Module):
    def __init__(self, output_dim=6,
                 feature_extraction_cnn='vgg',
                 feature_extraction_last_layer='',
                 return_correlation=False,
                 fr_feature_size=15,
                 fr_kernel_sizes=[7,5],
                 fr_channels=[128,64],
                 feature_self_matching=False,
                 normalize_features=True, normalize_matches=True,
                 batch_normalization=True,
                 train_fe=False,use_cuda=True):
#                 regressor_channels_1 = 128,
#                 regressor_channels_2 = 64):

        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.feature_self_matching = feature_self_matching
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)

        self.FeatureCorrelation = FeatureCorrelation(shape='3D',normalization=normalize_matches)


        self.FeatureRegression = FeatureRegression(output_dim,
                                                   use_cuda=self.use_cuda,
                                                   feature_size=fr_feature_size,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels,
                                                   batch_normalization=batch_normalization)


        self.ReLU = nn.ReLU(inplace=True)

    # used only for foward pass at eval and for training with strong supervision
    def forward(self, img1, img2):
        # feature extraction
        feature_A = self.FeatureExtraction(img1)
        feature_B = self.FeatureExtraction(img2)
        # feature correlation
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        # regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)

        if self.return_correlation:
            return (theta,correlation)
        else:
            return theta

class TwoStageCNNGeometric(CNNGeometric):
    def __init__(self,
                 fr_feature_size=15,
                 fr_kernel_sizes=[7,5],
                 fr_channels=[128,64],
                 feature_extraction_cnn='vgg',
                 feature_extraction_last_layer='',
                 return_correlation=False,
                 normalize_features=True,
                 normalize_matches=True,
                 batch_normalization=True,
                 train_fe=False,
                 use_cuda=True,
                 s1_output_dim=6,
                 s2_output_dim=18):

        super(TwoStageCNNGeometric, self).__init__(output_dim=s1_output_dim,
                                                   fr_feature_size=fr_feature_size,
                                                   fr_kernel_sizes=fr_kernel_sizes,
                                                   fr_channels=fr_channels,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   feature_extraction_last_layer=feature_extraction_last_layer,
                                                   return_correlation=return_correlation,
                                                   normalize_features=normalize_features,
                                                   normalize_matches=normalize_matches,
                                                   batch_normalization=batch_normalization,
                                                   train_fe=train_fe,
                                                   use_cuda=use_cuda)

        if s1_output_dim==6:
            self.geoTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
        else:
            tps_grid_size = np.sqrt(s2_output_dim/2)
            self.geoTnf = GeometricTnf(geometric_model='tps', tps_grid_size=tps_grid_size, use_cuda=use_cuda)

        self.FeatureRegression2 = FeatureRegression(output_dim=s2_output_dim,

                                                    use_cuda=use_cuda,
                                                    feature_size=fr_feature_size,
                                                    kernel_sizes=fr_kernel_sizes,
                                                    channels=fr_channels,
                                                    batch_normalization=batch_normalization)

    def forward(self, batch, f_src=None, f_tgt=None, use_theta_GT_aff=False):
        #===  STAGE 1 ===#
        if f_src is None and f_tgt is None:
            # feature extraction
            f_src = self.FeatureExtraction(batch['source_image'])
            f_tgt = self.FeatureExtraction(batch['target_image'])
        # feature correlation
        correlation_1 = self.FeatureCorrelation(f_src,f_tgt)
        # regression to tnf parameters theta
        theta_1 = self.FeatureRegression(correlation_1)

        #===  STAGE 2 ===#
        # warp image 1
        if use_theta_GT_aff==False:
            source_image_wrp = self.geoTnf(batch['source_image'],theta_1)
        else:
            source_image_wrp = self.geoTnf(batch['source_image'],batch['theta_GT_aff'])
        # feature extraction
        f_src_wrp = self.FeatureExtraction(source_image_wrp)
        # feature correlation
        correlation_2 = self.FeatureCorrelation(f_src_wrp,f_tgt)
        # regression to tnf parameters theta
        theta_2 = self.FeatureRegression2(correlation_2)

        if self.return_correlation:
            return (theta_1,theta_2,correlation_1,correlation_2)
        else:
            return (theta_1,theta_2)
