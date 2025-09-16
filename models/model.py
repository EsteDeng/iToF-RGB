import math
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from . import modules, encoder, decoder

class RGBD_Model(nn.Module):

    def __init__(self, resnet_depth=34, up_mode='bilinear', 
                       pretrained_img_encoder=True, 
                       resblock_decoder=False, out_size=(640,480), 
                       depth_res=False, out_act='tanh', 
                       ):
        '''
        resnet depth: 18, 34, 50, 101, 152
        mode: MonoRGB, StereoRGB, RGBD
        '''
        super().__init__()
        self.resnet_depth = resnet_depth
        self.up_mode = up_mode
        self.pretrained_img_encoder = pretrained_img_encoder
        self.resblock_decoder = resblock_decoder

        self.pretrained_tf = torchvision.transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.double_dim = False
        if pretrained_img_encoder:
            self.E_img = encoder.ResnetEncoder(resnet_depth, pretrained_img_encoder, 3)
            self.E_depth = encoder.ResnetEncoder(resnet_depth, False, 1)
            self.double_dim = True
        else:
            self.E_img = encoder.ResnetEncoder(resnet_depth, False, 4)
            self.E_depth = None
            
        self.D = decoder.MonoDepthDecoder(backbone_depth=resnet_depth, up_mode=up_mode,
                                      use_resblock=resblock_decoder, out_size=out_size, 
                                      depth_res=depth_res, out_act=out_act, double_dim=self.double_dim,
                                      )
                                    
    def forward(self, x_img, x_depth):
        if self.pretrained_img_encoder:
            x_img = self.pretrained_tf(x_img)
        else:
            x_img = x_img * 2 -1

        depth_feats = None
        if self.pretrained_img_encoder:
            img_feats = self.E_img(x_img)
            depth_feats = self.E_depth(x_depth * 2 - 1)
        else:
            img_feats = self.E_img(torch.cat([x_img, x_depth * 2 - 1], dim=1))

        depths = self.D(img_feats, depth_feats)
        
        return depths

