import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from . import modules

class MonoDepthDecoder(nn.Module):
    
    def __init__(self, backbone_depth=18, up_mode='nearest', 
                 use_resblock=False, out_size=(640,480), depth_res=False,
                 out_act='tanh', double_dim=False,):
        super().__init__()
        
        self.backbone_depth = backbone_depth
        self.up_mode = up_mode
        self.use_resblock = use_resblock
        self.depth_res = depth_res
        self.out_act = out_act
        self.double_dim = double_dim
        self.out_size = out_size

        self.up_times = 5
        self.shallow_resnet_dim_list = [512, 256, 128, 64, 64]
        self.deep_resnet_dim_list = [2048, 1024, 542, 256, 64]

        if backbone_depth<50:
            self.dim_list = self.shallow_resnet_dim_list
        else:
            self.dim_list = self.deep_resnet_dim_list


        if self.use_resblock:
            self.res_blocks = []
        self.feat_in_blocks = []
        self.depth_blocks = []

        self.feat_upample_layer = nn.Upsample(scale_factor=2, mode=up_mode)
            
        self.depth_upample_layer = nn.Upsample(size=out_size, mode=up_mode)

        for up_index in range(self.up_times):

            if up_index ==0:
                if self.double_dim:
                    in_dim = self.dim_list[up_index]*2
                else:
                    in_dim = self.dim_list[up_index]
                out_dim = self.dim_list[up_index+1]
            elif up_index ==4:
                if self.double_dim:
                    in_dim = self.dim_list[up_index]*3
                else:
                    in_dim = self.dim_list[up_index]*2
                out_dim = self.dim_list[up_index]
            else:
                if self.double_dim:
                    in_dim = self.dim_list[up_index]*3
                else:
                    in_dim = self.dim_list[up_index]*2
                out_dim = self.dim_list[up_index+1]

            self.feat_in_blocks +=[modules.Conv2dModule(in_dim, out_dim, 3,1,1, norm='none', act='elu')]
            if self.use_resblock:
                self.res_blocks += [modules.Resblock(out_dim, norm='bn', act='elu')]
            self.depth_blocks += [modules.Conv2dModule(out_dim, 1, 3,1,1, act=out_act)]
                
        if self.use_resblock:
            self.res_blocks = nn.ModuleList(self.res_blocks)
        self.feat_in_blocks = nn.ModuleList(self.feat_in_blocks)
        self.depth_blocks = nn.ModuleList(self.depth_blocks)


    def forward(self, x_feats, y_feats=None, depth_input=None):
        '''
        x_feats BxCxHxW
        depth_input: Bx1xHxW [0,1]
        '''


        depths = []
        for index in range(5):
            feat_cat = [x_feats[4-index]]
            if self.double_dim:
                feat_cat.append(y_feats[4-index])
            if index!=0:
                feat_cat.append(h)

            h = torch.cat(feat_cat, dim=1)
            
            h = self.feat_in_blocks[index](h)
            h = self.feat_upample_layer(h)

            if self.use_resblock:
                h = self.res_blocks[index](h)

            if index==0:
                
                depth_0 = self.depth_blocks[index](h) 
                
                if self.depth_res:
                    depth_last = depth_0
                depth_0 = self.depth_upample_layer(depth_0)
                if self.out_act == 'tanh':
                    depths.append((depth_0+1)/2)
                elif self.out_act == 'sigmoid':
                    depths.append(depth_0)
                    
            elif index==4:
                depth_4 = self.depth_blocks[index](h) 
                if self.depth_res:
                    depth_4 = depth_4 + self.feat_upample_layer(depth_last)
                depth = self.depth_upample_layer(depth_4)
                if self.out_act == 'tanh':
                    depths.append((depth+1)/2)
                elif self.out_act == 'sigmoid':
                    depths.append(depth)
            else:
                depth_i = self.depth_blocks[index](h)
                if self.depth_res:
                    depth_i = depth_i + self.feat_upample_layer(depth_last)
                    depth_last = depth_i
                depth = self.depth_upample_layer(depth_i)
                if self.out_act == 'tanh':
                    depths.append((depth+1)/2)
                elif self.out_act == 'sigmoid':
                    depths.append(depth)

        return depths
