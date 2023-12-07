# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init

# utils
import math
import os
import datetime
import numpy as np
import joblib

from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from collections import deque
import random
from torchvision import models
# from swin_transformer import SwinTransformer

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',}


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    if name == "nn":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes, kwargs.setdefault("dropout", False))
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "hamida":
        patch_size = kwargs.setdefault("patch_size", 11)
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.005)
        # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        optimizer = optim.Adam(model.parameters(),lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name =='dbda':
        patch_size = kwargs.setdefault("patch_size", 11)
        center_pixel = True
        model = DBDA(n_bands,n_classes)
        lr = kwargs.setdefault("learning_rate", 0.005)
        optimizer = optim.Adam(model.parameters(),lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name =='ssrn':
        patch_size = kwargs.setdefault("patch_size", 11)
        center_pixel = True
        model = SSRN(n_bands, n_classes)
        lr = kwargs.setdefault("learning_rate", 0.005)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
    elif name =='swt':
        patch_size = kwargs.setdefault("patch_size", 7)
        center_pixel = True
        model = SwinTransformer(img_size=7,
                                patch_size=4,
                                in_chans=144,#144
                                num_classes=n_classes,
                                embed_dim=48, #48 # 128
                                depths=[4, 2, 2, 2],
                                num_heads=[4, 4, 4, 4],
                                window_size=7,
                                mlp_ratio=4.0,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0,
                                drop_path_rate=0.5,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
        lr = kwargs.setdefault("learning_rate", 0.0005)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == "RNN":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = RNN(n_bands, n_classes)
        # From what I infer from the paper (Eq.7 and Algorithm 1), it is standard SGD with lr = 0.01
        lr = kwargs.setdefault("learning_rate", 0.0005)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == 'TGRS21':
        # here 20200401
        center_pixel = True
        kwargs.setdefault('patch_size', 11)
        kwargs.setdefault('epoch', 100)
        model = Network(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.005)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'dbma':
        patch_size = kwargs.setdefault("patch_size", 11)
        model = DBMA_network(n_bands, n_classes)
        lr = kwargs.setdefault("learning_rate", 0.005)
        center_pixel = True
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
    elif name == "hrwn":
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = True
        model = HRWN(n_bands, n_classes)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00005)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "lee":
        kwargs.setdefault("epoch", 200)
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = False
        model = LeeEtAl(n_bands, n_classes)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "chen":
        patch_size = kwargs.setdefault("patch_size", 27)
        center_pixel = True
        model = ChenEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.003)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 400)
        kwargs.setdefault("batch_size", 100)
    elif name == "li":
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = True
        model = LiEtAl(n_bands, n_classes, n_planes=16, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        )
        epoch = kwargs.setdefault("epoch", 200)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        # kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))
    elif name == "hu":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = HuEtAl(n_bands, n_classes)
        # From what I infer from the paper (Eq.7 and Algorithm 1), it is standard SGD with lr = 0.01
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "he":
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault("patch_size", 7)
        kwargs.setdefault("batch_size", 40)
        lr = kwargs.setdefault("learning_rate", 0.01)
        center_pixel = True
        model = HeEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "luo":
        # All  the  experiments  are  settled  by  the  learning  rate  of  0.1,
        # the  decay  term  of  0.09  and  batch  size  of  100.
        kwargs.setdefault("patch_size", 3)
        kwargs.setdefault("batch_size", 100)
        lr = kwargs.setdefault("learning_rate", 0.1)
        center_pixel = True
        model = LuoEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.09)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "sharma":
        # We train our S-CNN from scratch using stochastic gradient descent with
        # momentum set to 0.9, weight decay of 0.0005, and with a batch size
        # of 60.  We initialize an equal learning rate for all trainable layers
        # to 0.05, which is manually decreased by a factor of 10 when the validation
        # error stopped decreasing. Prior to the termination the learning rate was
        # reduced two times at 15th and 25th epoch. [...]
        # We trained the network for 30 epochs
        kwargs.setdefault("batch_size", 60)
        epoch = kwargs.setdefault("epoch", 30)
        lr = kwargs.setdefault("lr", 0.05)
        center_pixel = True
        # We assume patch_size = 64
        kwargs.setdefault("patch_size", 64)
        model = SharmaEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault(
            "scheduler",
            optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1
            ),
        )
    elif name == "liu":
        kwargs["supervision"] = "semi"
        # "The learning rate is set to 0.001 empirically. The number of epochs is set to be 40."
        kwargs.setdefault("epoch", 40)
        lr = kwargs.setdefault("lr", 0.001)
        center_pixel = True
        patch_size = kwargs.setdefault("patch_size", 9)
        model = LiuEtAl(n_bands, n_classes, patch_size)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # "The unsupervised cost is the squared error of the difference"
        criterion = (
            nn.CrossEntropyLoss(weight=kwargs["weights"]),
            lambda rec, data: F.mse_loss(
                rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()
            ),
        )
    elif name == "boulch":
        kwargs["supervision"] = "semi"
        kwargs.setdefault("patch_size", 1)
        kwargs.setdefault("epoch", 100)
        lr = kwargs.setdefault("lr", 0.001)
        center_pixel = True
        model = BoulchEtAl(n_bands, n_classes)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = (
            nn.CrossEntropyLoss(weight=kwargs["weights"]),
            lambda rec, data: F.mse_loss(rec, data.squeeze()),
        )
    elif name == "mou":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        kwargs.setdefault("epoch", 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault("lr", 1.0)
        model = MouEtAl(n_bands, n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])

    #########resnet###########
    elif name == "tbfc":
        # patch_size = kwargs.setdefault("patch_size", 27)
        center_pixel = True
        model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=True, **kwargs)
        lr = kwargs.setdefault("learning_rate", 0.1)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=epoch // 4, verbose=True
        ),
    )
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault("batch_size", 64)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs


class Baseline(nn.Module):
    """
    Baseline network
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=False, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2)) # kernel_size=(1, 2, 2)
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # if sub_sample:
        #     self.g = nn.Sequential(self.g, max_pool_layer)
        #     self.phi = nn.Sequential(self.phi, max_pool_layer)


    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)
        # print('x1', x.size())
        a = self.g(x)
        b = a.size(3)
        c = a.size(4)
        l = b*c

        g_x = self.g(x)
        g_x = g_x.permute(0, 3, 4, 1, 2).contiguous()
        g_x = g_x.view(batch_size, l, -1)

        theta_x = self.theta(x)
        theta_x = theta_x.permute(0, 3, 4, 1, 2).contiguous()
        theta_x = theta_x.view(batch_size, l, -1)

        phi_x = self.phi(x)
        phi_x = phi_x.permute(0, 3, 4, 1, 2).contiguous()
        phi_x = phi_x.view(batch_size, l, -1)
        phi_x = phi_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, x.size(2), *x.size()[3:])

        W_y = self.W(y)
        # z = W_y + x
        z = W_y

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer,)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()
    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class gelu(nn.Module):
    def __init__(self):
        super(gelu, self).__init__()
    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class gelu_new(nn.Module):
    def __init__(self):
        super(gelu_new, self).__init__()
        #Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
        #Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.sigmoid(x)


import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, Conv3d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width, channle = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*channle).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*channle)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*channle)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, channle = x.size()
        #print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) #形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        out = self.gamma*out + x  #C*H*W
        return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.input_channel = 20
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv3d(self.input_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class RNN(nn.Module):
    def __init__(self,  input_channels, n_classes, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_feature = 128
        self.timestep = 3
        self.input_channels = input_channels
        self.rnn = nn.LSTM(int(self.input_channels/self.timestep), self.hidden_feature, num_layers) # 使用两层 lstm
        self.classifier = nn.Linear(self.hidden_feature, n_classes) # 将最后一个 rnn 的输出使用全连接得到最后的分类结果

    def forward(self, x):
        '''
        x 大小为 (batch, 1, 28, 28)，所以我们需要将其转换成 RNN 的输入形式，即 (28, batch, 28)
        '''
        x = x[:,:,3,3]
        batch = x.shape[0]
        b = int(x.shape[1]/self.timestep)
        x1 = torch.zeros(batch,self.timestep,b).to('cuda')
        # x = x.unsqueeze(dim=2) # 去掉 (batch, 1, 28, 28) 中的 1，变成 (batch, 28, 28)
        x = x.squeeze()
        for j in range(0, self.timestep):
            x1[:,j,:] = x[:,j:j + (self.input_channels - 1) * self.timestep + 1:self.timestep]
        # x = x.permute(2,0,1) # 将最后一维放到第一维，变成 (28, batch, 28)
        x1 = x1.permute(1,0,2) # 将最后一维放到第一维，变成 (28, batch, 28)
        out, _ = self.rnn(x1) # 使用默认的隐藏状态，得到的 out 是 (28, batch, hidden_feature)
        out1 = torch.zeros( batch, self.hidden_feature).to('cuda')
        for j in range(0, self.timestep):
            out1 = out[j, :, :] + out1 # 取序列中的最后一个，大小是 (batch, hidden_feature)

        out1 = self.classifier(out1) # 得到分类结果

        return out1

class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1
            )
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0
            )
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        # self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = torch.unsqueeze(x,1)#[64 1 224 11 11]
        x = F.relu(self.conv1(x))#[64 20 222 9 9]
        x = self.pool1(x)#[64 20 111 9 9]
        x = F.relu(self.conv2(x))#[64 35 111 7 7]
        x = self.pool2(x)#[64 35 56 7 7]
        x = F.relu(self.conv3(x))#[64 35 56 7 7]
        x = F.relu(self.conv4(x))#[64 35 29 7 7]
        x = x.view(-1, self.features_size)#[64,49735]
        # x = self.dropout(x)
        x = self.fc(x)#[64 4]
        return x


class xie(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(xie, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1
            )
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0
            )
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        # self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x

class DBDA(nn.Module):
    def __init__(self, band, classes):
        super(DBDA, self).__init__()

        # spectral branch
        self.name = 'DBDA_MISH'
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))

        # Dense block
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new()
            # swish()
            mish()
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))  # kernel size随数据变化

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        self.conv25 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                      kernel_size=(3, 3, 2), stride=(1, 1, 1)),
            nn.Sigmoid()
        )

        self.batch_norm_spectral = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new(),
            # swish(),
            mish(),
            nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new(),
            # swish(),
            mish(),
            nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(120, classes)  # ,
            # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

        # fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        # spectral
        X = X.permute(0,1,3,4,2)
        x11 = self.conv11(X)
        # print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        # print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        # print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        # print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        # print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)

        # spatial
        # print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        # print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.batch_norm_spatial(x2)
        x2 = self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        # print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output

class SSRN(nn.Module):
    def __init__(self, band, classes):
        super(SSRN, self).__init__()
        self.name = 'SSRN'
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                               kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.res_net1 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net2 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(24, 24, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(24, 24, (3, 3, 1), (1, 1, 0))

        kernel_3d = math.ceil((band - 6) / 2)

        self.conv2 = nn.Conv3d(in_channels=24, out_channels=128, padding=(0, 0, 0),
                               kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0),
                               kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(24, classes)  # ,
            # nn.Softmax()
        )

    def forward(self, X):
        X = X.permute(0,1,3,4,2)
        x1 = self.batch_norm1(self.conv1(X))
        # print('x1', x1.shape)

        x2 = self.res_net1(x1)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        # print(x10.shape)
        return self.full_connection(x4)

class Network(nn.Module):
    def __init__(self, band, classes):
        super(Network, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(8, 1, 1), stride=(3, 1, 1), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.nl_1 = NONLocalBlock3D(in_channels=32)
        self.conv_2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=128, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.nl_2 = NONLocalBlock3D(in_channels=128)
        self.conv_3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.nl_3 = NONLocalBlock3D(in_channels=256)
        self.fc = nn.Sequential(
            nn.Linear(in_features=256*3*3*3, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=classes)
        )


    def forward(self, x):
        batch_size = x.size(0)
        # x = x.permute(0,1,3,4,2)
        feature_1 = self.conv_1(x)
        nl_feature_1 = self.nl_1(feature_1)

        feature_2 = self.conv_2(nl_feature_1)
        nl_feature_2 = self.nl_2(feature_2)

        # output = self.conv_3(nl_feature_2).view(batch_size, -1)
        feature_3 = self.conv_3(nl_feature_2)
        output = self.nl_3(feature_3).view(batch_size, -1)
        output = self.fc(output)

        return output

    def forward_with_nl_map(self, x):
        batch_size = x.size(0)

        feature_1 = self.conv_1(x)
        nl_feature_1, nl_map_1 = self.nl_1(feature_1, return_nl_map=True)

        feature_2 = self.conv_2(nl_feature_1)
        nl_feature_2, nl_map_2 = self.nl_2(feature_2, return_nl_map=True)

        # output = self.conv_3(nl_feature_2).view(batch_size, -1)
        feature_3 = self.conv_3(nl_feature_2)
        nl_feature_3, nl_map_3 = self.nl_3(feature_3, return_nl_map=True)
        output = nl_feature_3.view(batch_size, -1)
        output = self.fc(output)

        return output, [nl_map_1, nl_map_2, nl_map_3]


class DBMA_network(nn.Module):
    def __init__(self, band, classes):
        super(DBMA_network, self).__init__()

        # spectral branch
        self.name = 'DBMA'
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv3d(in_channels=48, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(72, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv3d(in_channels=72, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(96, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=96, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))  # kernel size随数据变化

        # 注意力机制模块

        # self.max_pooling1 = nn.MaxPool3d(kernel_size=(7, 7, 1))
        # self.avg_pooling1 = nn.AvgPool3d(kernel_size=(7, 7, 1))
        self.max_pooling1 = nn.AdaptiveAvgPool3d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool3d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv3d(in_channels=60, out_channels=30,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(in_channels=30, out_channels=60,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )
        # self.fc11 = Dense(30, activation=None, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
        # self.fc12 = Dense(60, activation=None, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

        self.activation1 = nn.Sigmoid()

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        # 注意力机制模块

        # self.max_pooling2 = nn.MaxPool3d(kernel_size=(1, 1, 60))
        # self.avg_pooling2 = nn.AvgPool3d(kernel_size=(1, 1, 60))
        # self.max_pooling2 = nn.AdaptiveAvgPool3d(1)
        # self.avg_pooling2 = nn.AdaptiveAvgPool3d(1)

        self.conv25 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                      kernel_size=(3, 3, 2), stride=(1, 1, 1)),
            nn.Sigmoid()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            nn.Linear(120, classes)  # ,
            # nn.Softmax()
        )

    def forward(self, X):
        # spectral
        x11 = self.conv11(X)
        # print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        # print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        # print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        # print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        # print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        x_max1 = self.max_pooling1(x16)
        x_avg1 = self.avg_pooling1(x16)
        # print('x_max1', x_max1.shape)

        # x_max1 = self.fc11(x_max1)
        # x_max1 = self.fc12(x_max1)
        #
        # x1_avg1 = self.fc11(x_avg1)
        # x1_avg1 = self.fc12(x_avg1)
        # print('x_max1', x_max1.shape)
        # x_max1 = x_max1.view(x_max1.size(0), -1)
        # x_avg1 = x_avg1.view(x_avg1.size(0), -1)
        # print('x_max1', x_max1.shape)
        x_max1 = self.shared_mlp(x_max1)
        x_avg1 = self.shared_mlp(x_avg1)
        # print('x_max1', x_max1.shape)
        x1 = torch.add(x_max1, x_avg1)
        x1 = self.activation1(x1)

        # x1 = x1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print('x1', x1.shape)
        # print('x16', x16.shape)

        # x1 = multiply([x1, x16])
        # x1 = self.activation1(x1)
        x1 = torch.mul(x1, x16)
        # print('x1', x1.shape)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        # print('x1', x1.shape)
        # x1 = Reshape(target_shape=(7, 7, 1, 60))(x1)
        # x1 = GlobalAveragePooling3D()(x1)

        # spatial
        # print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        # print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        # x_max2 = self.max_pooling2(x25)
        # x_avg2 = self.avg_pooling2(x25)
        # x_avg2 = x_avg2.permute(0, 4, 2, 3, 1)
        x_avg2 = torch.mean(x25, dim=1, keepdim=True)
        x_max2, _ = torch.max(x25, dim=1, keepdim=True)
        # print('x_avg2', x_avg2.shape)

        x2 = torch.cat((x_max2, x_avg2), dim=-1)
        x2 = self.conv25(x2)
        # print('x2', x2.shape)
        # print('x25', x25.shape)

        x2 = torch.mul(x2, x25)
        # print('x2', x2.shape)
        x2 = self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)
        # x2 = Reshape(target_shape=(7, 7, 1, 60))(x2)
        # x2 = GlobalAveragePooling3D()(x2)

        # print('x1', x1.shape)
        # print('x2', x2.shape)

        x_pre = torch.cat((x1, x2), dim=1)
        # print('x_pre', x_pre.shape)
        x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output


class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0
        )

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        return x


class ChenEtAl(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """

    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=27, n_planes=32):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x


class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes, (3, 3, 3), padding=(1, 0, 0))
        # self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LuoEtAl(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=90):
        super(LuoEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9, 1, 1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HRWN(nn.Module):
    def __init__(self, input_channel, n_class):
        super(HRWN, self).__init__()
        self.n_class = n_class
        filters=[32,64,100,200,256]
        self.conv0_spat=nn.Conv2d(input_channel, filters[2], kernel_size=3, stride=1, padding=1)
        self.relu0 = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(filters[2])

        self.conv1_spat=nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(filters[2])

        self.conv2_spat=nn.Conv2d(filters[2], filters[3], kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(filters[3])

        self.conv3_spat=nn.Conv2d(filters[3], filters[3], kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(filters[3])

        self.max2d = nn.MaxPool2d((2,2))


        self.fc1 = nn.Linear(800,1024)  ###
        self.relu6 = nn.ReLU(inplace=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024,512)
        self.dropout1 = nn.Dropout(p=0.4)

        self.conv0 = nn.Conv1d(input_channel, out_channels=64, kernel_size=11)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2)

        self.max1d = nn.MaxPool1d(2)


        self.dropout2 = nn.Dropout(p=0.5)
        self.fc_l = nn.Linear(1024,self.n_class)

    def forward(self, x):
        ####smc#####

        p = x[:,:,2,2]
        p = torch.unsqueeze(p,2)
        x = self.conv0_spat(x)
        x = self.relu0(x)
        x = self.bn0(x)
        x = self.conv1_spat(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2_spat(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv3_spat(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.max2d(x)
        shape_b = x.shape
        x = x.view(shape_b[0],-1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout1(x)
        x = torch.flatten(x)

        p = self.conv0(p)
        p = self.bn4(p)
        p = self.relu4(p)
        p = self.conv1(p)
        p = self.relu5(p)
        p = self.max1d(p)
        p = torch.flatten(p)

        m = torch.cat([x,p])
        m = self.dropout2(m)
        m = self.fc_l(m)
        return m

class SharmaEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=64):
        super(SharmaEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv1 = nn.Conv3d(1, 96, (input_channels, 6, 6), stride=(1, 2, 2))
        self.conv1_bn = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        #  256 kernels of size 3x3x256 with a stride of 2 pixels
        self.conv2 = nn.Conv3d(1, 256, (96, 3, 3), stride=(1, 2, 2))
        self.conv2_bn = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # 512 kernels of size 3x3x512 with a stride of 1 pixel
        self.conv3 = nn.Conv3d(1, 512, (256, 3, 3), stride=(1, 1, 1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = self.pool1(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = self.pool2(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv3(x))
            print(x.size())
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LiuEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(LiuEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_bn = nn.BatchNorm2d(80)

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        # Decoder
        self.fc1_dec = nn.Linear(self.features_sizes[2], self.features_sizes[2])
        self.fc1_dec_bn = nn.BatchNorm1d(self.features_sizes[2])
        self.fc2_dec = nn.Linear(self.features_sizes[2], self.features_sizes[1])
        self.fc2_dec_bn = nn.BatchNorm1d(self.features_sizes[1])
        self.fc3_dec = nn.Linear(self.features_sizes[1], self.features_sizes[0])
        self.fc3_dec_bn = nn.BatchNorm1d(self.features_sizes[0])
        self.fc4_dec = nn.Linear(self.features_sizes[0], input_channels)

        self.apply(self.weight_init)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        _, c, w, h = x.size()
        size0 = c * w * h

        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h

        x = self.conv1_bn(x)
        _, c, w, h = x.size()
        size2 = c * w * h

        return size0, size1, size2

    def forward(self, x):
        x = x.squeeze()
        x_conv1 = self.conv1_bn(self.conv1(x))
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = F.relu(x).view(-1, self.features_sizes[2])
        x = x_enc

        x_classif = self.fc_enc(x)

        # x = F.relu(self.fc1_dec_bn(self.fc1_dec(x) + x_enc))
        x = F.relu(self.fc1_dec(x))
        x = F.relu(
            self.fc2_dec_bn(self.fc2_dec(x) + x_pool1.view(-1, self.features_sizes[1]))
        )
        x = F.relu(
            self.fc3_dec_bn(self.fc3_dec(x) + x_conv1.view(-1, self.features_sizes[0]))
        )
        x = self.fc4_dec(x)
        return x_classif, x


class BoulchEtAl(nn.Module):
    """
    Autoencodeurs pour la visualisation d'images hyperspectrales
    A.Boulch, N. Audebert, D. Dubucq
    GRETSI 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, planes=16):
        super(BoulchEtAl, self).__init__()
        self.input_channels = input_channels
        self.aux_loss_weight = 0.1

        encoder_modules = []
        n = input_channels
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            print(x.size())
            while n > 1:
                print("---------- {} ---------".format(n))
                if n == input_channels:
                    p1, p2 = 1, 2 * planes
                elif n == input_channels // 2:
                    p1, p2 = 2 * planes, planes
                else:
                    p1, p2 = planes, planes
                encoder_modules.append(nn.Conv1d(p1, p2, 3, padding=1))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.MaxPool1d(2))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.ReLU(inplace=True))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.BatchNorm1d(p2))
                x = encoder_modules[-1](x)
                print(x.size())
                n = n // 2

            encoder_modules.append(nn.Conv1d(planes, 3, 3, padding=1))
        encoder_modules.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_modules)
        self.features_sizes = self._get_sizes()

        self.classifier = nn.Linear(self.features_sizes, n_classes)
        self.regressor = nn.Linear(self.features_sizes, input_channels)
        self.apply(self.weight_init)

    def _get_sizes(self):
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            x = self.encoder(x)
            _, c, w = x.size()
        return c * w

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(-1, self.features_sizes)
        x_classif = self.classifier(x)
        x = self.regressor(x)
        return x_classif, x


class MouEtAl(nn.Module):
    """
    Deep recurrent neural networks for hyperspectral image classification
    Lichao Mou, Pedram Ghamisi, Xiao Xang Zhu
    https://ieeexplore.ieee.org/document/7914752/
    """

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data, -0.1, 0.1)
            init.uniform_(m.bias.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(MouEtAl, self).__init__()
        self.input_channels = input_channels
        self.gru = nn.GRU(1, 64, 1, bidirectional=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d(64 * input_channels)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(64 * input_channels, n_classes)

    def forward(self, x):
        x = x.squeeze()
        x = x.unsqueeze(0)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1, 0)
        x = self.gru(x)[0]
        # x is in C, N, 64, we permute back
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0), -1)
        x = self.gru_bn(x)
        x = self.tanh(x)
        x = self.fc(x)
        return x


_CURRENT_STORAGE_STACK = []


class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])


def fc_init_weights(m):
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch/49433937#49433937
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


class ResNet50(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        resnet = models.resnet50(pretrained=False)
        #self.conv1 = nn.Conv2d(224, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.fc = nn.Linear(2048, 4)
        self.fc.apply(fc_init_weights)
        self.feature_store = Store(4, 20)
        self.means = [None for _ in range(4)]
        self.hingeloss = nn.HingeEmbeddingLoss(2)
        self.margin = 10.0
        self.clustering_momentum = 0.99

    def get_event_storage(self, ):
        """
        Returns:
            The :class:`EventStorage` object that's currently being used.
            Throws an error if no :class:`EventStorage` is currently enabled.
        """
        assert len(
            _CURRENT_STORAGE_STACK
        ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
        return _CURRENT_STORAGE_STACK[-1]

    def clstr_loss_l2_cdist(self, input_features, gt_classes):
        """
        Get the foreground input_features, generate distributions for the class,
        get probability of each feature from each distribution;
        Compute loss: if belonging to a class -> likelihood should be higher
                      else -> lower
        :param input_features:
        :param proposals:
        :return:
        """
        fg_features = input_features
        classes = gt_classes
        # fg_features = F.normalize(fg_features, dim=0)
        # fg_features = self.ae_model.encoder(fg_features)

        all_means = self.means
        for item in all_means:
            if item != None:
                length = item.shape
                break

        for i, item in enumerate(all_means):
            if item == None:
                all_means[i] = torch.zeros((length))

        distances = torch.cdist(fg_features, torch.stack(all_means).cuda(), p=self.margin)
        labels = []

        for index, feature in enumerate(fg_features):
            for cls_index, mu in enumerate(self.means):
                if mu is not None and feature is not None:
                    if classes[index] == cls_index:
                        labels.append(1)
                    else:
                        labels.append(-1)
                else:
                    labels.append(0)

        loss = self.hingeloss(distances, torch.tensor(labels).reshape((-1, 4)).cuda())

        return loss

    def get_clustering_loss(self, input_features, proposals, i):
        s_iter = i
        c_loss = 0
        if s_iter == 0:
            items = self.feature_store.retrieve(-1)
            for index, item in enumerate(items):
                if len(item) == 0:
                    self.means[index] = None
                else:
                    mu = torch.tensor(item).mean(dim=0)
                    self.means[index] = mu
            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
            # Freeze the parameters when clustering starts
            # for param in self.ae_model.parameters():
            #     param.requires_grad = False
        elif s_iter > 0:
            if s_iter % 10 == 0:
                # Compute new MUs
                items = self.feature_store.retrieve(-1)
                new_means = [None for _ in range(4)]
                for index, item in enumerate(items):
                    if len(item) == 0:
                        new_means[index] = None
                    else:
                        new_means[index] = torch.tensor(item).mean(dim=0)
                # Update the MUs
                for i, mean in enumerate(self.means):
                    if (mean) is not None and new_means[i] is not None:
                        self.means[i] = self.clustering_momentum * mean + (1 - self.clustering_momentum) * new_means[i]

            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
        return c_loss

    def forward(self, x, gt, i, mode='test'):

        x = self.encoder(x)
        # print(x.shape)
        x1 = x.view(x.size(0), -1)


        x = self.fc(x1)

        #x = F.normalize(x)  #

        if mode == 'train':
            self.feature_store.add(x1, gt)
            cluster_loss = self.get_clustering_loss(x1, gt, i)
        else:
            cluster_loss = 0
        return x, cluster_loss


import torch
import glob
import torch.nn as nn
import torch.nn.functional as F


class CoTNetLayer(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            # 通过K*K的卷积提取上下文信息，视作输入X的静态上下文表达
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False),  # 1*1的卷积进行Value的编码
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(  # 通过连续两个1*1的卷积计算注意力矩阵
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),  # 输入concat后的特征矩阵 Channel = 2*C
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1, stride=1)  # out: H * W * (K*K*C)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # shape：bs,c,h,w  提取静态上下文信息得到key
        v = self.value_embed(x).view(bs, c, -1)  # shape：bs,c,h*w  得到value编码

        y = torch.cat([k1, x], dim=1)  # shape：bs,2c,h,w  Key与Query在channel维度上进行拼接进行拼接
        att = self.attention_embed(y)  # shape：bs,c*k*k,h,w  计算注意力矩阵
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # shape：bs,c,h*w  求平均降低维度
        k2 = F.softmax(att, dim=-1) * v  # 对每一个H*w进行softmax后
        k2 = k2.view(bs, c, h, w)

        return k1 + k2  # 注意力融合

class CoTNetLayer_pool(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.poolq = nn.Conv2d(dim, dim, 3, 2, 1)
        self.key_embed = nn.Sequential(
            # 通过K*K的卷积提取上下文信息，视作输入X的静态上下文表达
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.poolk = nn.Conv2d(dim, dim, 3, 2, 1)
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False),  # 1*1的卷积进行Value的编码
            nn.BatchNorm2d(dim)
        )
        self.poolv = nn.Conv2d(dim, dim, 3, 2, 1)
        factor = 4
        self.attention_embed = nn.Sequential(  # 通过连续两个1*1的卷积计算注意力矩阵
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),  # 输入concat后的特征矩阵 Channel = 2*C
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1, stride=1)  # out: H * W * (K*K*C)
        )

    def forward(self, x):

        q = self.poolq(x)
        bs, c, h, w = q.shape
        k1 = self.key_embed(x)  # shape：bs,c,h,w  提取静态上下文信息得到key
        k1 = self.poolk(k1)

        v = self.value_embed(x)  # shape：bs,c,h*w  得到value编码
        v = self.poolv(v).view(bs, c, -1)

        y = torch.cat([k1, q], dim=1)  # shape：bs,2c,h,w  Key与Query在channel维度上进行拼接进行拼接
        att = self.attention_embed(y)  # shape：bs,c*k*k,h,w  计算注意力矩阵
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # shape：bs,c,h*w  求平均降低维度
        k2 = F.softmax(att, dim=-1) * v  # 对每一个H*w进行softmax后
        k2 = k2.view(bs, c, h, w)

        return k1 + k2  # 注意力融合


class SPC31(nn.Module):
    def __init__(self, outplane, kernel_size=[7, 1, 1], padding=[3, 0, 0]):
        super(SPC31, self).__init__()
        self.poolx = nn.Conv2d(outplane, outplane, 3, 2, 1)
        self.cotnet = CoTNetLayer_pool(outplane)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

        self.convm0 = nn.Conv3d(1, outplane, kernel_size=kernel_size, padding=padding)
        self.convm1 = nn.Conv3d(1, outplane, kernel_size=kernel_size, padding=padding)  # generate mask0
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x, identity=None):
        if identity is None:
            identity = self.poolx(x)  # NCHW

        x = self.cotnet(x)
        x = self.bn1(x)
        x = self.relu(x)

        n, c, h, w = x.size()

        mask0 = self.convm0(x.unsqueeze(1)).squeeze(2)  # NCHW ==> NDHW
        mask0 = torch.softmax(mask0.view(n, -1, h * w), -1)
        mask0 = mask0.view(n, -1, h, w)
        _, d, _, _ = mask0.size()

        mask1 = self.convm1(x.unsqueeze(1)).squeeze(2)  # NDHW
        mask1 = torch.softmax(mask1.view(n, -1, h * w), -1)
        mask1 = mask1.view(n, -1, h, w)

        fk = torch.einsum('ndhw,nchw->ncd', mask0, x)  # NCD

        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask1)  # NCDHW

        out = F.leaky_relu(out)
        out = out.sum(2)

        out = out + mask0  # + identity

        out = out + identity

        out = self.bn2(out.view(n, -1, h, w))

        return out  # NCHW


class SPC32(nn.Module):
    def __init__(self, outplane,  kernel_size=[7, 1, 1], padding=[3, 0, 0]):
        super(SPC32, self).__init__()
        self.cotnet = CoTNetLayer(outplane)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

        self.convm0 = nn.Conv3d(1, outplane, kernel_size=kernel_size, padding=padding)
        self.convm1 = nn.Conv3d(1, outplane, kernel_size=kernel_size, padding=padding)  # generate mask0
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x, identity=None):
        if identity is None:
            identity = x  # NCHW

        x = self.cotnet(x)
        x = self.bn1(x)
        x = self.relu(x)

        n, c, h, w = x.size()

        mask0 = self.convm0(x.unsqueeze(1)).squeeze(2)  # NCHW ==> NDHW
        mask0 = torch.softmax(mask0.view(n, -1, h * w), -1)
        mask0 = mask0.view(n, -1, h, w)
        _, d, _, _ = mask0.size()

        mask1 = self.convm1(x.unsqueeze(1)).squeeze(2)  # NDHW
        mask1 = torch.softmax(mask1.view(n, -1, h * w), -1)
        mask1 = mask1.view(n, -1, h, w)

        fk = torch.einsum('ndhw,nchw->ncd', mask0, x)  # NCD

        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask1)  # NCDHW

        out = F.leaky_relu(out)
        out = out.sum(2)

        out = out + mask0  # + identity

        out = out + identity

        out = self.bn2(out.view(n, -1, h, w))

        return out  # NCHW

class SPC33(nn.Module):
    def __init__(self, outplane, kernel_size=[7, 1, 1], padding=[3, 0, 0]):
        super(SPC33, self).__init__()
        self.poolx = nn.Conv2d(outplane, outplane, 3, 2, 1)
        self.layer1= nn.Conv2d(outplane, outplane, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

        self.layer2= nn.Conv2d(outplane, outplane, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x, identity=None):
        if identity is None:
            identity = self.poolx(x)  # NCHW

        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = x + identity
        x = self.bn2(x)

        return x # NCHW

class SPC34(nn.Module):
    def __init__(self, outplane,  kernel_size=[7, 1, 1], padding=[3, 0, 0]):
        super(SPC34, self).__init__()
        self.layer1= nn.Conv2d(outplane, outplane, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

        self.layer2= nn.Conv2d(outplane, outplane, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x, identity=None):
        if identity is None:
            identity = x  # NCHW

        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = x + identity
        x = self.bn2(x)

        return x  # NCHW

class SSNet_AEAE(nn.Module):
    def __init__(self, num_classes=4, msize=25, inter_size=100):
        super(SSNet_AEAE, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(224, inter_size, 3, 1, 1, bias=False),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(inter_size), )
        # nn.LeakyReLU())

        #self.pool1 = nn.Conv2d(inter_size, inter_size, 3, 2, 1)
        self.layer2 = SPC33(outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0])
        self.layer3 = SPC34(outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0])

        self.layer4 = nn.Conv2d(inter_size, msize, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(msize)

        #self.pool2 = nn.Conv2d(msize, msize, 3, 2, 1)
        self.layer5 = SPC33(outplane=msize, kernel_size=[msize, 1, 1], padding=[0, 0, 0])
        self.layer6 = SPC34(outplane=msize, kernel_size=[msize, 1, 1], padding=[0, 0, 0])

        self.fc = nn.Linear(msize*9, num_classes)

    def forward(self, x):
        x = np.squeeze(x)
        x = self.layer1(x)#[64 49 11 11]

        #x = self.pool1(x)
        x = self.layer2(x)#[64 49 11 11]
        x = self.layer3(x)#[64 49 11 11]


        x = self.bn4(F.leaky_relu(self.layer4(x)))#[64 18 11 11]

        #x = self.pool2(x)
        x = self.layer5(x)
        x = self.layer6(x)
        n, c, h, w = x.size()

        x = x.reshape(-1, c*h*w)
        x = self.fc(x)

        return x


class SpatAttn(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, ratio=8):
        super(SpatAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()  # BxCxHxW
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxHWxC
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW
        energy = torch.bmm(proj_query, proj_key)  # BxHWxHW, attention maps
        attention = self.softmax(energy)  # BxHWxHW, normalized attn maps
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxHW
        out = out.view(m_batchsize, C, height, width)  # BxCxHxW

        out = self.gamma * out + x
        return out


class SpatAttn_(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, ratio=8):
        super(SpatAttn_, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.Sequential(nn.ReLU(),
                                nn.BatchNorm2d(in_dim))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()  # BxCxHxW
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxHWxC
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW
        energy = torch.bmm(proj_query, proj_key)  # BxHWxHW, attention maps
        attention = self.softmax(energy)  # BxHWxHW, normalized attn maps
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # BxCxHW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxHW
        out = out.view(m_batchsize, C, height, width)  # BxCxHxW

        out = self.gamma * out  # + x
        return self.bn(out)


class SARes(nn.Module):
    def __init__(self, in_dim, ratio=8, resin=False):
        super(SARes, self).__init__()

        if resin:
            self.sa1 = SpatAttn(in_dim, ratio)
            self.sa2 = SpatAttn(in_dim, ratio)
        else:
            self.sa1 = SpatAttn_(in_dim, ratio)
            self.sa2 = SpatAttn_(in_dim, ratio)

    def forward(self, x):
        identity = x
        x = self.sa1(x)
        x = self.sa2(x)

        return F.relu(x + identity)


# class SPC3(nn.Module):
#     def __init__(self, msize=24, outplane=49, kernel_size=[7, 1, 1], stride=[1, 1, 1], padding=[3, 0, 0], spa_size=9,
#                  bias=True):
#         super(SPC3, self).__init__()
#
#         self.convm0 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)  # generate mask0
#         self.convm1 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)  # generate mask1
#
#         self.bn2 = nn.BatchNorm2d(outplane)
#
#     def forward(self, x):
#         identity = x  # NCHW
#         # n,c,h,w = identity.size()
#
#         mask0 = self.convm0(x.unsqueeze(1)).squeeze(2)  # NCHW ==> NDHW
#         n, _, h, w = mask0.size()
#
#         mask0 = torch.softmax(mask0.view(n, -1, h * w), -1)
#         mask0 = mask0.view(n, -1, h, w)
#         _, d, _, _ = mask0.size()
#
#         mask1 = self.convm0(x.unsqueeze(1)).squeeze(2)  # NDHW
#         mask1 = torch.softmax(mask1.view(n, -1, h * w), -1)
#         mask1 = mask1.view(n, -1, h, w)
#         # print(mask1.size())
#
#         fk = torch.einsum('ndhw,nchw->ncd', mask0, x)  # NCD
#
#         out = torch.einsum('ncd,ndhw->ncdhw', fk, mask1)  # NCDHW
#
#         out = F.leaky_relu(out)
#         out = out.sum(2)
#
#         out = out + identity
#
#         out = self.bn2(out.view(n, -1, h, w))
#
#         return out  # NCHW
#
#
# class SPC32(nn.Module):
#     def __init__(self, msize=24, outplane=49, kernel_size=[7, 1, 1], stride=[1, 1, 1], padding=[3, 0, 0], spa_size=9,
#                  bias=True):
#         super(SPC32, self).__init__()
#
#         self.convm0 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)  # generate mask0
#         self.bn1 = nn.BatchNorm2d(outplane)
#
#         self.convm2 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)  # generate mask2
#         self.bn2 = nn.BatchNorm2d(outplane)
#
#     def forward(self, x, identity=None):
#         if identity is None:
#             identity = x  # NCHW
#         n, c, h, w = identity.size()
#
#         mask0 = self.convm0(x.unsqueeze(1)).squeeze(2)  # NCHW ==> NDHW
#         mask0 = torch.softmax(mask0.view(n, -1, h * w), -1)
#         mask0 = mask0.view(n, -1, h, w)
#         _, d, _, _ = mask0.size()
#
#         fk = torch.einsum('ndhw,nchw->ncd', mask0, x)  # NCD
#
#         out = torch.einsum('ncd,ndhw->ncdhw', fk, mask0)  # NCDHW
#
#         out = F.leaky_relu(out)
#         out = out.sum(2)
#
#         out = out  # + identity
#
#         out0 = self.bn1(out.view(n, -1, h, w))
#
#         mask2 = self.convm2(out0.unsqueeze(1)).squeeze(2)  # NCHW ==> NDHW
#         mask2 = torch.softmax(mask2.view(n, -1, h * w), -1)
#         mask2 = mask2.view(n, -1, h, w)
#
#         fk = torch.einsum('ndhw,nchw->ncd', mask2, x)  # NCD
#
#         out = torch.einsum('ncd,ndhw->ncdhw', fk, mask2)  # NCDHW
#
#         out = F.leaky_relu(out)
#         out = out.sum(2)
#
#         out = out + identity
#
#         out = self.bn2(out.view(n, -1, h, w))
#
#         return out  # NCHW
#
#
# class SPCModule(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(SPCModule, self).__init__()
#
#         self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False)
#         # self.bn = nn.BatchNorm3d(out_channels)
#
#     def forward(self, input):
#         out = self.s1(input)
#
#         return out
#
#
# class SPCModuleIN(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(SPCModuleIN, self).__init__()
#
#         self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=False)
#         # self.bn = nn.BatchNorm3d(out_channels)
#
#     def forward(self, input):
#         input = input.unsqueeze(1)
#
#         out = self.s1(input)
#
#         return out.squeeze(1)
#
#
# class SPAModuleIN(nn.Module):
#     def __init__(self, in_channels, out_channels, k=49, bias=True):
#         super(SPAModuleIN, self).__init__()
#
#         # print('k=',k)
#         self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k, 3, 3), bias=False)
#         # self.bn = nn.BatchNorm2d(out_channels)
#
#     def forward(self, input):
#         # print(input.size())
#         out = self.s1(input)
#         out = out.squeeze(2)
#         # print(out.size)
#
#         return out
#
#
# class ResSPC(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(ResSPC, self).__init__()
#
#         self.spc1 = nn.Sequential(
#             nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
#             nn.LeakyReLU(inplace=True),
#             nn.BatchNorm3d(in_channels), )
#
#         self.spc2 = nn.Sequential(
#             nn.Conv3d(in_channels, in_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
#             nn.LeakyReLU(inplace=True), )
#
#         self.bn2 = nn.BatchNorm3d(out_channels)
#
#     def forward(self, input):
#         out = self.spc1(input)
#         out = self.bn2(self.spc2(out))
#
#         return F.leaky_relu(out + input)
#
#
# class ResSPA(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(ResSPA, self).__init__()
#
#         self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#                                   nn.LeakyReLU(inplace=True),
#                                   nn.BatchNorm2d(in_channels), )
#
#         self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#                                   nn.LeakyReLU(inplace=True), )
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, input):
#         out = self.spa1(input)
#         out = self.bn2(self.spa2(out))
#
#         return F.leaky_relu(out + input)
#
#
# class SSRN(nn.Module):
#     def __init__(self, num_classes=4, k=109):
#         super(SSRN, self).__init__()
#
#         self.layer1 = SPCModuleIN(1, 28)
#         # self.bn1 = nn.BatchNorm3d(28)
#
#         self.layer2 = ResSPC(28, 28)
#
#         self.layer3 = ResSPC(28, 28)
#
#         # self.layer31 = AKM(28, 28, [97,1,1])
#         self.layer4 = SPAModuleIN(28, 28, k=k)
#         self.bn4 = nn.BatchNorm2d(28)
#
#         self.layer5 = ResSPA(28, 28)
#         self.layer6 = ResSPA(28, 28)
#
#         self.fc = nn.Linear(28, num_classes)
#
#     def forward(self, x):
#         x = F.leaky_relu(self.layer1(x))  # self.bn1(F.leaky_relu(self.layer1(x)))
#         # print(x.size())
#         x = self.layer2(x)
#         x = self.layer3(x)
#         # x = self.layer31(x)
#         x = F.leaky_relu(self.layer4(x))
#         x = self.bn4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#
#         x = F.avg_pool2d(x, x.size()[-1])
#         x = self.fc(x.squeeze())
#
#         return x
#
#
# class SSNet(nn.Module):
#     def __init__(self, num_classes=4, msize=18, inter_size=49):
#         super(SSNet, self).__init__()
#
#         self.layer1 = nn.Sequential(nn.Conv2d(224, inter_size, 1, bias=False),
#                                     nn.LeakyReLU(),
#                                     nn.BatchNorm2d(inter_size), )
#         # nn.LeakyReLU())
#
#         self.layer2 = SARes(inter_size, ratio=8)  # ResSPA(inter_size, inter_size)
#         self.layer3 = SPC32(msize, outplane=inter_size, kernel_size=[inter_size, 1, 1], padding=[0, 0, 0])
#
#         self.layer4 = nn.Conv2d(inter_size, msize, kernel_size=1)
#         self.bn4 = nn.BatchNorm2d(msize)
#
#         self.layer5 = SARes(msize, ratio=8)  # ResSPA(msize, msize)
#         self.layer6 = SPC32(msize, outplane=msize, kernel_size=[msize, 1, 1], padding=[0, 0, 0])
#
#         self.fc = nn.Linear(msize, num_classes)
#
#     def forward(self, x):
#         n, c, h, w = x.size()
#
#         x = self.layer1(x)
#
#         x = self.layer2(x)
#         x = self.layer3(x)
#
#
#         x = self.bn4(F.leaky_relu(self.layer4(x)))
#         x = self.layer5(x)
#         x = self.layer6(x)
#
#         x = F.avg_pool2d(x, x.size()[-1])
#         x = self.fc(x.squeeze())
#
#         return x


def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cpu"),
    display=None,
    val_loader=None,
    supervision="full",
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    save_epoch = epoch // 20 if epoch > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        # Run the training loop for one epoch     ####hrwn-data1_pixel
        for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            # Load the data into the GPU if required
            # data = torch.squeeze(data,1)
            data, target = data.to(device), target.to(device)
            # data = torch.squeeze(data)
            optimizer.zero_grad()
            if supervision == "full":
                ########swt###########
                data = torch.squeeze(data)
                ######################
                output = net(data)
                loss = criterion(output, target)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter : iter_],
                    win=loss_win,
                    update=update,
                    opts={
                        "title": "Training loss",
                        "xlabel": "Iterations",
                        "ylabel": "Loss",
                    },
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(
                        Y=np.array(val_accuracies),
                        X=np.arange(len(val_accuracies)),
                        win=val_win,
                        opts={
                            "title": "Validation accuracy",
                            "xlabel": "Epochs",
                            "ylabel": "Accuracy",
                        },
                    )
            iter_ += 1
            del (data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e % save_epoch == 0:
            save_model(
                net,
                camel_to_snake(str(net.__class__.__name__)),
                data_loader.dataset.name,
                epoch=e,
                metric=abs(metric),
            )


def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
    """
    Using strftime in case it triggers exceptions on windows 10 system
    """
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = time_str + "_epoch{epoch}_{metric:.2f}".format(
            **kwargs
        )
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        filename = time_str
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + ".pkl")


def test(net, img):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = 11
    center_pixel = True
    batch_size, device = 100, 'cuda'
    n_classes = 3

    kwargs = {
        "step": 1,
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            ########swt###########
            data = torch.squeeze(data)
            ######################
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs

def test_clusloss(net, img, test_gt, i):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = 11
    center_pixel = True
    batch_size, device = 100, 'cuda'
    n_classes =4

    kwargs = {
        "step": 1,
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            ########swt###########
            data = torch.squeeze(data)
            ######################
            #output,_ = net(data,data,1)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs

def val(net, data_loader, device="cpu", supervision="full"):
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == "full":
                ########swt###########
                data = torch.squeeze(data)
                ######################
                output = net(data)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total


if __name__=='__main__':
    print(get_model('tbfc'))