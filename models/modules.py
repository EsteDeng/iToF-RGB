import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class AdaptiveInstanceNorm(nn.Module):
    '''
    Module Interface for AdaIN(Adaptive Instance Normalization)    
    '''
    def calc_mean_std(self, x, eps=1e-5):
        x = x.reshape(x.size(0), -1)
        return x.mean(1, keepdim=True), x.std(1, keepdim=True)


    def adaptive_instance_norm(self, feat, adaptor, eps=1e-5):
        size = feat.size()
        feat_mean, feat_std = self.calc_mean_std(feat)
        adaptive_mean, adaptive_std = self.calc_mean_std(adaptor)

        normalized_feat = (feat - feat_mean) / feat_std
        adapted_feat = (normalized_feat*adaptive_std)+adaptive_mean
        return adapted_feat
    
    def __init__(self, eps=1e-5):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = eps
        
    def forward(self, x, y):
        '''
        x: feature for normalization
        y: feature for adaption
        '''
        return self.adaptive_instance_norm(x,y, eps=self.eps)
    
class ConditionalBatchNorm2d(nn.Module):
    '''
    code credit: https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    '''
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = y.chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class Conv2dModule(nn.Module):
    def __init__(self, in_dim , out_dim, k, s, p, norm='none', act='relu', leak=0.2, sn=False, transpose=False):
        '''
        norm : bn in |ln cbn adain adalin cln
        act: relu lrelu selu, elu
        '''
        super().__init__()
        self.transpose = transpose
        if self.transpose:
            self.conv = nn.ConvTranspose2d(in_dim, out_dim, k, s, p)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, k, s, p)

        if sn:
            self.conv = spectral_norm(self.conv)

        self.conditional = False

        # normalization
        if norm =='none':
            self.norm = None
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(out_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_dim)
        elif norm == 'ln':
            raise NotDebugYet
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.conditional = True

        # conditional normalization
        if norm == 'cbn':
            self.norm = ConditionalBatchNorm2d(out_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm()
        elif norm == 'adalin':
            raise NotImplementYet
        elif norm == 'cln':
            raise NotImplementYet

        if act=='none':
            self.act = None
        elif act=='relu':
            self.act = nn.ReLU()
        elif act=='lrelu':
            self.act = nn.LeakyReLU(leak)
        elif act=='selu':
            self.act = nn.SELU()
        elif act=='elu':
            self.act = nn.ELU()
        elif act=='tanh':
            self.act = nn.Tanh()
        elif act=='sigmoid':
            self.act = nn.Sigmoid()
        elif act=='softplus':
            self.act = nn.Softplus()
        elif act=='relu6':
            self.act = nn.ReLU6()

    def forward(self,x, c=None):
        h = self.conv(x)
        if self.norm:
            if self.conditional:
                h = self.norm(h, c)
            else:
                h = self.norm(h)

        if self.act:
            h = self.act(h)

        return h

class Resblock(nn.Module):
    def __init__(self, dim, norm='none', act='relu', sn=False):
        super().__init__()
        
        self.conv_0 = Conv2dModule(dim, dim, 3,1,1, norm=norm, act=act, sn=sn)
        self.conv_1 = Conv2dModule(dim, dim, 3,1,1, norm=norm, act='none', sn=sn)

    def forward(self, x, c=None):
        h = self.conv_0(x,c)
        h = self.conv_1(h,c)

        return x+h
