import os, torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
from torchvision import transforms
import scipy.io as sio
import numpy as np
class ToF_FlyingTings3D_UW_Simple_Dataset(torch.utils.data.Dataset):

    height = 480
    width = 640

    def __init__(self, root, mode='train'):
        super().__init__()

        self.root = root
        self.mode = mode
        
        if self.mode == 'train' or self.mode == 'val' or self.mode == 'test':
            list_path = os.path.join(self.root, f'{self.mode}_list.txt')
            with open(list_path, 'r') as f:
                self.filenames = f.read().split('\n')
                self.filenames = [f for f in self.filenames if f != '']
                self.filenames = [f.replace(' ', '') for f in self.filenames]

        else:
            raise NotImplementedError


    def __getitem__(self, index):
        #load img
        rgb_path = os.path.join(self.root, 'gt_depth_rgb', f'{self.filenames[index]}_rgb.png')
        gt_depth_path = os.path.join(self.root, 'gt_depth_rgb', f'{self.filenames[index]}_gt_depth.mat')
        ndepth_path = os.path.join(self.root, 'nToF', f'{self.filenames[index]}_noisy_depth.mat')
        conf_path = os.path.join(self.root, 'gt_depth_rgb', f'{self.filenames[index]}_gt_conf.mat')

        rgb = Image.open(rgb_path).convert('RGB')
        rgb = TF.to_tensor(rgb)

        gt_depth = torch.tensor(sio.loadmat(gt_depth_path)['gt_depth']).unsqueeze(0).float()
        ndepth = torch.tensor(sio.loadmat(ndepth_path)['ndepth']).unsqueeze(0).float()
        conf = torch.tensor(sio.loadmat(conf_path)['conf']).unsqueeze(0).float()
    
        
        
        return gt_depth, rgb, ndepth, conf
    

    def __len__(self):
        return len(self.filenames)