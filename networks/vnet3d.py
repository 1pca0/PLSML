# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
# https://github.com/huangzhii/FCN-3D-pytorch/blob/master/main3d.py
import torch
import torch.nn as nn
import os
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        # self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        # self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)
        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(self.num_features)

        self.relu1 = ELUCons(elu, self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.outchans = outChans

        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)


        self.sigmoid = nn.Sigmoid()
        self.upsampling4 = nn.Upsample(scale_factor=2)
        self.upsampling8 = nn.Upsample(scale_factor=4)
        self.dsconv6 = nn.Conv3d(128, 1, 3, 1, 1)  # deep supervision
        self.dsconv7 = nn.Conv3d(64, 1, 3, 1, 1)  # deep supervision
        self.dsconv8 = nn.Conv3d(32, 1, 3, 1, 1)  # deep supervision
        # self.dsconv6 = nn.Conv3d(128, 1, 5, 1, 2)  # deep supervision
        # self.dsconv7 = nn.Conv3d(64, 1, 5, 1, 2)  # deep supervision
        # self.dsconv8 = nn.Conv3d(32, 1, 5, 1, 2)  # deep supervision
        self.upsampling = nn.Upsample(scale_factor=1)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)

        out = self.ops(xcat)
        if self.outchans==128:
            # print(f"神监督前out.shape{out.shape}")
            ds_6 = self.sigmoid((self.upsampling8(self.dsconv6(out))))
            # print(f"ds6{ds_6.shape}")
            out = self.relu2(torch.add(out, xcat))
            return [out,ds_6]
        elif self.outchans==64:
            ds_7 = self.sigmoid((self.upsampling4(self.dsconv7(out))))
            out = self.relu2(torch.add(out, xcat))
            return [out,ds_7]
        elif self.outchans==32:
            ds_8 = self.sigmoid((self.upsampling(self.dsconv8(out))))
            out = self.relu2(torch.add(out, xcat))
            return [out, ds_8]
        else:
            return out
        #return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels, classes, elu,coord=True):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self._coord = coord
        if self._coord:
            num_channel_coord = 3
        else:
            num_channel_coord =0
        # self.conv1 = nn.Conv3d(in_channels+num_channel_coord, classes, kernel_size=5, padding=2)
        self.conv1 = nn.Conv3d(in_channels + num_channel_coord, classes, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

    def forward(self, x,coordmap =None):
        if self._coord and (coordmap is not None):
            x = torch.cat([x,coordmap],dim=1)
        else:
            x =x
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class VNet(nn.Module):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, elu=True, in_channels=1, classes=1,coord =True):
        super(VNet, self).__init__()
        self.classes = classes
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, elu=elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=False)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=False)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=False)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=False)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, classes, elu,coord)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x,coord):
        out16 = self.in_tr(x)

        # print(f"out16.shape{out16.shape}")
        out32 = self.down_tr32(out16)

        # print(f"out32.shape{out32.shape}")
        out64 = self.down_tr64(out32)

        #print(f"out64.shape{out64.shape}")
        out128 = self.down_tr128(out64)

        #print(f"out128.shape{out128.shape}")
        out256 = self.down_tr256(out128)

        #print(f"out256.shape{out256.shape}")
        out = self.up_tr256(out256, out128)

        #print(f"out.shape{out.shape}")
        out,ds_6 = self.up_tr128(out, out64)

        #print(f"out1.shape{out.shape}")
        out,ds_7 = self.up_tr64(out, out32)

        #print(f"out2.shape{out.shape}")
        out,ds_8 = self.up_tr32(out, out16)
        conv9 = out
        #print(f"out3.shape{out.shape}")
        out = self.out_tr(out,coord)
        # out = self.sigmoid(out)
        #print(f"out4.shape{out.shape}")
        return [out,conv9,ds_6]

smooth = 1
def dice_loss(pred, target):
	"""
	DSC loss
	: param pred: input prediction
	: param target: input target
	"""
	iflat = pred.view(-1)
	tflat = target.view(-1)
	intersection = torch.sum((iflat * tflat))
	return 1. - ((2. * intersection + smooth)/(torch.sum(iflat) + torch.sum(tflat) + smooth))



# net = VNet(1,1)
# xx = torch.randn(4,1,128,128,128)
# coord = torch.randn(4,3,128,128,128)
# yy = net(xx,coord)
# print(f"ds6.shape{yy[1].shape}")
# print(f"ds7.shape{yy[3].shape}")
# print(f"ds8.shape{yy[2].shape}")
# # # label = torch.randn(2,1,128,128,128)>0
# # # loss = dice_loss(yy[0],label)
# # # loss2 = dice_loss(yy[1],label)
# # # loss3 = dice_loss(yy[2],label)
# # # loss4 = dice_loss(yy[3],label)
# # # print(loss)
# print(yy[0].shape)








   