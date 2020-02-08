from layer import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler


class CLS(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(CLS, self).__init__()
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        # Classification Network
        self.conv1 = Conv2d(self.nch_in, 10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = Pooling2d(pool=2, type='max')
        self.relu1 = ReLU(0.0)

        self.conv2 = Conv2d(10,          20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(0.5)
        self.pool2 = Pooling2d(pool=2, type='max')
        self.relu2 = ReLU(0.0)

        self.fc1 = Linear(nch_in=320, nch_out=50)
        self.relu_fc1 = ReLU(0.0)
        self.drop_fc1 = nn.Dropout2d(0.5)
        self.fc2 = Linear(nch_in=50, nch_out=10)

    def forward(self, x):
        # perform the usual forward pass
        x = self.relu1(self.pool1(self.conv1(x)))

        x = self.relu2(self.pool2(self.drop2(self.conv2(x))))

        x = x.view(-1, 320)

        x = self.drop_fc1(self.relu_fc1(self.fc1(x)))
        x = self.fc2(x)

        x = torch.log_softmax(x, dim=1)

        return x


class STN(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(STN, self).__init__()
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        # Localization Network part of Spatial Transform Network (STN)
        localization = []
        localization += [Conv2d(nch_in=self.nch_in, nch_out=8,  kernel_size=7, stride=1, padding=0, bias=True)]
        localization += [Pooling2d(pool=2, type='max')]
        localization += [ReLU(0.0)]

        localization += [Conv2d(nch_in=8,           nch_out=10, kernel_size=5, stride=1, padding=0, bias=True)]
        localization += [Pooling2d(pool=2, type='max')]
        localization += [ReLU(0.0)]

        # localization += [Conv2d(nch_in=self.nch_in, nch_out=self.nch_ker, kernel_size=3, stride=1, padding=1, bias=False)]
        # localization += [ReLU(0.0)]
        #
        # localization += [Conv2d(nch_in=self.nch_ker, nch_out=self.nch_ker, kernel_size=3, stride=1, padding=1, bias=False)]
        # localization += [ReLU(0.0)]
        # localization += [Pooling2d(pool=2, type='max')]
        #
        # localization += [Conv2d(nch_in=self.nch_ker, nch_out=self.nch_ker, kernel_size=3, stride=1, padding=1, bias=False)]
        # localization += [ReLU(0.0)]
        # localization += [Pooling2d(pool=2, type='max')]
        #
        # localization += [Conv2d(nch_in=self.nch_ker, nch_out=self.nch_ker, kernel_size=3, stride=1, padding=1, bias=False)]
        # localization += [ReLU(0.0)]
        # localization += [Pooling2d(pool=2, type='max')]

        self.localization = nn.Sequential(*localization)

        # Regression Network part of Spatial Transform Network (STN)
        affinement = []
        affinement += [Linear(nch_in=10 * 3 * 3, nch_out=32)]
        affinement += [ReLU(0.0)]
        affinement += [Linear(nch_in=32, nch_out=2 * 3)]

        self.affinement = nn.Sequential(*affinement)

    def forward(self, x):
        # transform the input
        x_loc = self.localization(x)
        x_loc = x_loc.view(-1, 10 * 3 * 3)
        theta = self.affinement(x_loc)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

