import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from collections import OrderedDict
from md.mdpytorch.init.kaiming_init import kaiming_weight_init
from md.mdpytorch.module.vbnet_inputblock import InputBlock
from md.mdpytorch.module.vbnet_downblock import DownBlock


def parameters_init(net):
    net.apply(kaiming_weight_init)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias), self.weight

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def refine_cams(cam_original, image_shape, using_sigmoid=True, cam_w=100, cam_sigma=0.4):
    cam_original = F.interpolate(
        cam_original, image_shape, mode="trilinear", align_corners=True
    )
    B, C, D, H, W = cam_original.size()
    cams = []
    for idx in range(C):
        cam = cam_original[:, idx, :, :, :]
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        norm = cam_max - cam_min
        norm[norm == 0] = 1e-5
        cam = (cam - cam_min) / norm
        cam = cam.view(B, D, H, W).unsqueeze(1)
        cams.append(cam)
    cams = torch.cat(cams, dim=1)
    if using_sigmoid:
        cams = torch.sigmoid(cam_w * (cams - cam_sigma))
    return cams


class ClassificationNet(nn.Module):

    def __init__(self, in_channels, class_num, input_size):
        super(ClassificationNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1)
        self.down_64 = DownBlock(32, 2)
        self.down_128 = DownBlock(64, 3)
        self.down_256 = DownBlock(128, 3)

        # 加入GAP
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.custom_fc = Linear(256, class_num, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, out_DeepFeature=False, cam_w=100, cam_sigma=0.4):

        _, _, D, H, W = input.size()

        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)

        # 采用 GAP 方式展开
        out = self.avgpool(out256)
        out = out.view(out.size(0), -1)
        deep_features = out.view(out.size(0), -1)

        out, fc_w = self.custom_fc(out)
        out = self.softmax(out)
        cam_classes = self.relu(
            F.conv3d(out256, fc_w.detach().unsqueeze(2).unsqueeze(3).unsqueeze(4), bias=None, stride=1, padding=0))
        cam_classes_refined = refine_cams(cam_classes, (D, H, W), using_sigmoid=False, cam_w=cam_w, cam_sigma=cam_sigma)

        # if out_DeepFeature:
        #     return out, deep_features
        # else:
        #     return out, cam_classes_refined

        return out, cam_classes_refined

    @staticmethod
    def max_stride():
        return 16
