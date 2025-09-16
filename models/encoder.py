import math
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from . import modules
import torchvision

class ResnetEncoder(nn.Module):
    def __init__(self, num_layers=50, pretrained=True, in_dim=3):
        '''
        torchvision
        '''
        super().__init__()
        # out name
        # fc: resnet, inception, googlenet, shufflenet, resnext50_32x4d, wide_resnet50_2
        # classifier: alexnet, vgg, squeezenet, mobilenet, mnasnet, classifier

        self.num_layers = num_layers
        self.pretrained = pretrained
        
        exec('self.model = torchvision.models.resnet{}(pretrained=pretrained)'.format(self.num_layers), 
             {'self':self,'torchvision': torchvision, 'pretrained': self.pretrained})
        
        if in_dim==3:
            input_block = [self.model.conv1, self.model.bn1, self.model.relu]
        else:
            self.conv_in = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.conv_in.weight, mode='fan_out', nonlinearity='relu')
            input_block = [self.conv_in, self.model.bn1, self.model.relu]
        self.input_block = nn.Sequential(*input_block)
        module_list = []
        module_list += [nn.Sequential(self.model.maxpool, self.model.layer1)]
        module_list += [self.model.layer2]
        module_list += [self.model.layer3]
        module_list += [self.model.layer4]
        self.model = nn.ModuleList(module_list)

    def forward(self, x):
        h = self.input_block(x)
        feats = [h]
        for module in self.model:
            h = module(h)
            feats.append(h)
        return feats