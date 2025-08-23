#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:34:22 2020

@author: wangwu
"""


from email.mime import base
import torch
import torch.nn as nn
from einops import rearrange
import scipy.io as sio
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
from timm.models.layers import trunc_normal_, DropPath
from math import exp
from torch.autograd import Variable
from IBP_pavia import GS_Attention
from network_swin2sr import BasicLayer
import os
from loss_vif import Sobelxy

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x    

class ConvNext(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Res(nn.Module):
    def __init__(self, n_feat):
        super(Res, self).__init__()
        self.main = nn.Sequential(
            #nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(n_feat),
            nn.ReLU(True),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True))
    def forward(self, x):
        return self.main(x)+x

class LP(nn.Module):
    def __init__(self, num_spectral):
        super(LP, self).__init__()
        
        self.num_spectral = num_spectral
        self.pad = nn.ReflectionPad2d(1)
        self.proj_in = nn.Conv2d(num_spectral, num_spectral, 3, 1, 0, bias=True)
        
        self.pool2 = torch.nn.MaxPool2d(2,stride=2)
        
        self.fe8 = Res(num_spectral)
        self.fe4 = Former(num_spectral)
        self.fe2 = Former(num_spectral)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.in_ln = nn.LayerNorm(num_spectral)
        self.out_ln = nn.LayerNorm(num_spectral)
        
    def forward(self, I):
        I = self.proj_in(self.pad(I))
        
        I2 = self.pool2(I)
        I2 = self.fe2(I2)
        h1 = I - self.up2(I2)

        I4 = self.pool2(I2)
        I4 = self.fe4(I4)
        h2 = I2 - self.up2(I4)

        I8 = self.pool2(I4)
        I8 = self.fe8(I8)
        h3 = I4 - self.up2(I8)
        
        return I8,h1,h2,h3


class GS_S_Attention(nn.Module):
    def __init__(self, base_filter,num_spectral):
        super(GS_S_Attention, self).__init__()
        
        self.base_filter = base_filter
        self.pad = nn.ReflectionPad2d(1)
        self.proj_in = nn.Conv2d(base_filter+1, base_filter, 3, 1, 1, bias=True)
        
        self.pool2 = torch.nn.MaxPool2d(3,stride=2)
        
        self.fe8 = ConvNext(base_filter)
        self.fe4 = ConvNext(base_filter)
        self.fe2 = ConvNext(base_filter)

        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.tran = nn.Conv2d(base_filter*4, base_filter, 3, 1, 1, bias=True)
        self.proj_out1 = nn.Conv2d(base_filter, 2, 3, 1, 1, bias=True)
        #self.proj_out2 = nn.Conv2d(num_spectral, 1, 3, 1, 0, bias=True)
    
        self.glo = BasicLayer(dim=base_filter,
                                     input_resolution=(128, 128),
                                     depth=2,
                                     num_heads=6,
                                     window_size=8,
                                     mlp_ratio=3,
                                     qkv_bias=True,
                                     drop=0, attn_drop=0,
                                     drop_path=0,
                                     norm_layer=nn.LayerNorm,
                                     downsample=None,
                                     use_checkpoint=False)
        self.in_ln = nn.LayerNorm(base_filter)
        self.out_ln = nn.LayerNorm(base_filter)
        
    def forward(self, I):
        I = self.proj_in(I)
        B,C,H,W = I.shape
        
        I2 = self.pool2(self.pad(I))
        I4 = self.pool2(self.pad(I2))
        I8 = self.pool2(self.pad(I4))
        
        
        I8 = self.fe8(I8)
        I8 = self.up8(I8)
        I4 = self.fe4(I4)
        I2 = self.fe2(I2)
        I4 = self.up4(I4)
        I2 = self.up2(I2)
        
        I = self.tran(torch.cat((I8,I4,I2,I),dim=1))
        
        I = rearrange(I, 'B C H W -> B (H W) C')
        I = self.out_ln(self.glo(self.in_ln(I),(H,W)))
        I = rearrange(I, 'B (H W) C-> B C H W',H=H,W=W)
        saliency = self.proj_out1(I)
        #detail = self.proj_out2(self.pad(I))
        I = F.softmax(saliency,dim=1)
        I1 = I[:,1,:,:].unsqueeze(1)
        I2 = I[:,0,:,:].unsqueeze(1)
        
        #I1 = torch.sigmoid(saliency)
        
        return I1

class Cell_Former_4070(nn.Module):
    def __init__(self, base_filter,num_spectral=1, window_size=8, depth=2):
        super(Cell_Former_4070, self).__init__()

        self.in_conv = nn.Conv2d(base_filter, base_filter, 3, 1, 1, bias=False)
        self.out_conv = nn.Conv2d(base_filter, base_filter, 3, 1, 1, bias=False)
        self.gs = GS_S_Attention(base_filter,num_spectral)
        # swin block
        self.glo = BasicLayer(dim=base_filter,
                                     input_resolution=(128, 128),
                                     depth=depth,
                                     num_heads=6,
                                     window_size=window_size,
                                     mlp_ratio=3,
                                     qkv_bias=True,
                                     drop=0, attn_drop=0,
                                     drop_path=0,
                                     norm_layer=nn.LayerNorm,
                                     downsample=None,
                                     use_checkpoint=False)
        
        self.in_ln = nn.LayerNorm(base_filter)
        self.out_ln = nn.LayerNorm(base_filter)
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(True)
    
        self.base_filter = base_filter

    def forward(self, X, Z, m):
        B,C,H,W = X.shape
        dZ = self.relu(Z - X)
        
        I = self.in_conv(dZ)
        I = rearrange(I, 'B C H W -> B (H W) C')
        I = self.out_ln(self.glo(self.in_ln(I),(H,W)))
        I = rearrange(I, 'B (H W) C-> B C H W',H=H,W=W)
        I = self.out_conv(I)
        m = self.gs(torch.cat((dZ,m),dim=1))
        X = X + I*m 
        #X = X + I
        return X, m

class Former(nn.Module):
    def __init__(self, base_filter, window_size=2, depth=2):
        super(Former, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.in_conv = nn.Conv2d(base_filter, base_filter, 3, 1, 0, bias=False)
        self.out_conv = nn.Conv2d(base_filter, base_filter, 3, 1, 0, bias=False)
        self.g = BasicLayer(dim=base_filter,
                                     input_resolution=(128, 128),
                                     depth=depth,
                                     num_heads=6,
                                     window_size=window_size,
                                     mlp_ratio=3,
                                     qkv_bias=True,
                                     drop=0, attn_drop=0,
                                     drop_path=0,
                                     norm_layer=nn.LayerNorm,
                                     downsample=None,
                                     use_checkpoint=False)
        self.in_ln = nn.LayerNorm(base_filter)
        self.out_ln = nn.LayerNorm(base_filter)
# 
    def forward(self, I):
        B,C,H,W = I.shape
        I = self.in_conv(self.pad(I))
        I = rearrange(I, 'B C H W -> B (H W) C')
        I = self.out_ln(self.g(self.in_ln(I),(H,W)))
        I = rearrange(I, 'B (H W) C-> B C H W',H=H,W=W)
        I = self.out_conv(self.pad(I))
        return I

class Cell_Former(nn.Module):
    def __init__(self, base_filter, window_size=8, depth=2):
        super(Cell_Former, self).__init__()

        self.in_conv_i1 = nn.Conv2d(1, base_filter, 3, 1, 0, bias=False)
        self.in_conv_i2 = nn.Conv2d(1, base_filter, 3, 1, 0, bias=False)
        
        self.se_atten = GS_S_Attention(base_filter)
        self.trans = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(base_filter*2, base_filter, 3, 1, 0, bias=False))
        self.g1 = Former(base_filter, window_size=window_size, depth=2)
        self.g2 = Former(base_filter, window_size=window_size, depth=2)
        
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(True)
        
# 
    def forward(self, I, I1, I2):
        I1 = self.in_conv_i1(self.pad(I1))
        I2 = self.in_conv_i2(self.pad(I2))
        
        dI2 = self.relu(I2 - I)
        dI1 = self.relu(I1 - I)
        m1, m2 = self.se_atten(self.trans(torch.cat((dI1,dI2),dim=1)))
        I = I + m1*self.g1(dI1) + m2*self.g1(dI2)

        return I,I

class Cell_Former1(nn.Module):
    def __init__(self, base_filter, window_size=8, depth=2):
        super(Cell_Former1, self).__init__()

        self.in_conv_i1 = nn.Conv2d(1, base_filter, 3, 1, 0, bias=False)
        self.in_conv_i2 = nn.Conv2d(1, base_filter, 3, 1, 0, bias=False)
        
        self.se_atten = GS_S_Attention(base_filter)
        self.trans = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(base_filter*2, base_filter, 3, 1, 0, bias=False))
        self.g1 = Former(base_filter, window_size=window_size, depth=2)
        self.g2 = Former(base_filter, window_size=window_size, depth=2)
        
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(True)
        
# 
    def forward(self, I, I1, I2):
        I1 = self.in_conv_i1(self.pad(I1))
        I2 = self.in_conv_i2(self.pad(I2))
        
        dI2 = self.relu(I2 - I)
        dI1 = self.relu(I1 - I)
        
        m1, m2 = self.se_atten(self.trans(torch.cat((dI1,dI2),dim=1)))
        I = I + m1*(self.g1(I1)-self.g1(I)) + m2*(self.g2(dI2) - self.g2(I))

        return I

class New_Trans(nn.Module):
    def __init__(self, num_channel=1, base_filter=48, num_spectral=1):
        super(New_Trans, self).__init__()

        self.in_conv_i1 = nn.Conv2d(1, base_filter, 3, 1, 0, bias=False)
        self.in_conv_i2 = nn.Conv2d(1, base_filter, 3, 1, 0, bias=False)
        self.se_atten = GS_S_Attention(base_filter)
        
        self.atten_t1 = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(base_filter, base_filter, 3, 1, 0, bias=False),
                                      nn.Sigmoid())
        self.atten_t2 = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(base_filter, base_filter, 3, 1, 0, bias=False),
                                      nn.Sigmoid())
        self.trans = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(base_filter*2, base_filter, 3, 1, 0, bias=False))
        self.t = Former(base_filter, window_size=2, depth=2)
        self.t1 = nn.Sequential(nn.ReflectionPad2d(1),
                                nn.Conv2d(base_filter, base_filter, 3, 1, 0, bias=False),
                                nn.LeakyReLU(inplace=True),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(base_filter, base_filter, 3, 1, 0, bias=False))
        self.t2 = nn.Sequential(nn.ReflectionPad2d(1),
                                nn.Conv2d(base_filter, base_filter, 3, 1, 0, bias=False),
                                nn.LeakyReLU(inplace=True),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(base_filter, base_filter, 3, 1, 0, bias=False))
        self.pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(base_filter, 1, 3, 1, 0, bias=False)
        
# 
    def forward(self, I1, I2):
        I1 = self.in_conv_i1(self.pad(I1))
        I2 = self.in_conv_i2(self.pad(I2))
        
        m1, m2 = self.se_atten(self.trans(torch.cat((I1,I2),dim=1)))
        tI1 = self.t(I1)
        tI2 = self.t(I2)
        I = m1*tI1 + m2*tI2 + self.atten_t1(self.t1(I1)-tI1) + self.atten_t2(self.t2(I2)-tI2)
        I = self.out_conv(self.pad(I))
        return I

class UF_Net(nn.Module):
    def __init__(self, base_filter, num_spectral):
        super(UF_Net, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(base_filter, num_spectral, 3, 1, 0, bias=False)
        
        self.encoder1 = nn.Sequential(#nn.ReflectionPad2d(1),
                                     nn.Conv2d(num_spectral, base_filter, 3, 1, 1, bias=False),
                                     Res(base_filter),
                                     Res(base_filter))
        self.encoder2 = nn.Sequential(#nn.ReflectionPad2d(1),
                                     nn.Conv2d(num_spectral, base_filter, 3, 1, 1, bias=False),
                                     Res(base_filter),
                                     Res(base_filter))
        self.decoder = nn.Sequential(Res(base_filter),
                                     Res(base_filter),
                                     Res(base_filter),
                                     #nn.ReflectionPad2d(1),
                                     nn.Conv2d(base_filter, num_spectral, 3, 1, 1, bias=False))
                                     #nn.Tanh())
        self.fuser = Unfolding_Net_V2(base_filter,num_spectral)
        #self.fuser = Vanilla_Net(base_filter)
        self.grad = Sobelxy()

    def forward(self, I1, I2):
        B,C,H,W = I1.shape
        mask = torch.zeros([B,1,H,W]).cuda()
        #pre_mask = torch.zeros([B,2,H,W]).cuda()
        F1 = self.encoder1(I1)
        F2 = self.encoder2(I2)
        out_I1 = self.decoder(F1)
        out_I2 = self.decoder(F2)
        
        F,mask  = self.fuser(F1,F2,mask)
    
        I = self.decoder(F)
        I2_grad = self.grad(I2)
        I1_grad = self.grad(I1)
        grad = self.grad(I)
        #grad = torch.max(I1_grad,I2_grad)
        #grad = torch.max(grad,f_grad)
        return I,out_I1,out_I2,grad

class UF_Net1(nn.Module):
    def __init__(self, base_filter, num_spectral):
        super(UF_Net1, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(base_filter, num_spectral, 3, 1, 0, bias=False)
        
        self.encoder1 = nn.Sequential(#nn.ReflectionPad2d(1),
                                     nn.Conv2d(num_spectral, base_filter, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(base_filter),
                                     nn.ReLU(True),
                                     Res(base_filter),
                                     Res(base_filter))
        self.encoder2 = nn.Sequential(#nn.ReflectionPad2d(1),
                                     nn.Conv2d(num_spectral, base_filter, 3, 1, 1, bias=False),
                                     Res(base_filter),
                                     Res(base_filter))
        self.decoder = nn.Sequential(Res(base_filter),
                                     Res(base_filter),
                                     Res(base_filter),
                                     #nn.ReflectionPad2d(1),
                                     nn.Conv2d(base_filter, num_spectral, 3, 1, 1, bias=False))
        self.fuser1 = Unfolding_Net_V2(base_filter,num_spectral)
        self.fuser2 = Unfolding_Net_V2(base_filter,num_spectral)
        #self.fuser = Vanilla_Net(base_filter)
        self.grad = Sobelxy()

    def forward(self, I1, I2):
        B,C,H,W = I1.shape
        mask = torch.zeros([B,1,H,W]).cuda()
        F1 = self.encoder1(I1)
        F2 = self.encoder1(I2)
        out_I1 = self.decoder(F1)
        out_I2 = self.decoder(F2)
        
        F  = self.fuser1(F1,F2,mask) + self.fuser2(F2,F1,mask)
    
        I = self.decoder(F)
        I2_grad = self.grad(I2)
        I1_grad = self.grad(I1)
        grad = self.grad(I)
        #grad = torch.max(I1_grad,I2_grad)
        #grad = torch.max(grad,f_grad)
        return I,out_I1,out_I2,grad

class Vanilla_Net(nn.Module):
    def __init__(self, base_filter):
        super(Vanilla_Net, self).__init__()

        self.trans = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(base_filter*2, base_filter, 3, 1, 0, bias=False))
        self.fuse = Former(base_filter,depth=6)
        
        
    def forward(self, Y, Z,mask):
        X = self.fuse(self.trans(torch.cat((Y,Z),dim=1)))
        return X

class Unfolding_Net_V2(nn.Module):
    def __init__(self, base_filter,num_spectral):
        super(Unfolding_Net_V2, self).__init__()

        self.u1 = Cell_Former_4070(base_filter,num_spectral, window_size=2)
        self.u2 = Cell_Former_4070(base_filter,num_spectral, window_size=2)
        self.u3 = Cell_Former_4070(base_filter,num_spectral, window_size=2)
        
    def forward(self, Y, Z,mask):
        
        X,mask1 = self.u1(Y, Z, mask)
        X,mask2 = self.u2(X, Z, mask1)
        X,mask3 = self.u3(X, Z, mask2)
    
        return X,mask1

class YDTR_Like(nn.Module):
    def __init__(self, num_spectral,base_filter):
        super(YDTR_Like, self).__init__()

        self.in_conv1 = nn.Sequential(nn.Conv2d(num_spectral, base_filter, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(base_filter),
                                      nn.ReLU(True),
                                      Former(base_filter),
                                      ConvNext(base_filter))
        #self.in_conv2 = nn.Sequential(nn.Conv2d(num_spectral, base_filter, 3, 1, 1, bias=False),
        #                              nn.BatchNorm2d(base_filter),
        #                             Res(base_filter),
        #                             Res(base_filter))
        self.out_conv = nn.Sequential(nn.Conv2d(base_filter, base_filter, 3, 1, 1, bias=False),
                                     Former(base_filter),
                                     ConvNext(base_filter),
                                     nn.Conv2d(base_filter, num_spectral, 3, 1, 1, bias=False),
                                     nn.Tanh(),
                                     )
        
    def forward(self, i1,i2):
        f1 = self.in_conv1(i1)
        f2 = self.in_conv1(i2)
        x = self.out_conv(f1+f2)
        return x,f1,f2,f1

class Color_Net(nn.Module):
    def __init__(self, num_spectral,base_filter):
        super(Color_Net, self).__init__()

        self.in_conv = nn.Conv2d(num_spectral*2+1, base_filter, 3, 1, 1, bias=False)
        self.out_conv = nn.Conv2d(base_filter, num_spectral, 3, 1, 1, bias=False)
        fuse = [ConvNext(base_filter) \
                for _ in range(6)]
        self.fuse = nn.Sequential(*fuse)
        
    def forward(self, x):
        x = self.out_conv(self.fuse(self.in_conv(x)))
        return x

