import warnings
warnings.simplefilter("ignore")
import os, pickle
import sys
sys.path.append("./MiDaS")
import configparser
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from MiDaS import hubconf
from tqdm import tqdm, trange
import kornia as K

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

EPS = 1e-8

class BasicTrainer(nn.Module): 
    DEFAULT_MAX_MAE = float('inf') 

    def _smooth_loss(self, depth, image, beta=1.0):
        """
        Calculate the image-edge-aware second-order smoothness loss for flo 
        modified from https://github.com/lelimite4444/BridgeDepthFlow/blob/14183b99830e1f41e774c0d43fdb058d07f2a397/utils/utils.py#L60
        """
        img_grad_x, img_grad_y = self._gradient(image)
        weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
        weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

        dx, dy = self._gradient(depth)

        dx2, dxdy = self._gradient(dx)
        dydx, dy2 = self._gradient(dy)
        del dxdy, dydx

        return (torch.mean(beta*weights_x[:,:, :, 1:]*torch.abs(dx2)) + torch.mean(beta*weights_y[:, :, 1:, :]*torch.abs(dy2))) / 2.0

    def compute_normals(self, depth):
        """
        输入:
        depth: [B, 1, H, W] tensor, 深度图

        返回:
        normals: [B, 3, H, W] tensor, 单位法线 (x, y, z)
        """
        # Sobel kernel
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=depth.dtype, device=depth.device).view(1,1,3,3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=depth.dtype, device=depth.device).view(1,1,3,3) / 8.0

        dzdx = F.conv2d(depth, sobel_x, padding=1)
        dzdy = F.conv2d(depth, sobel_y, padding=1)

        # 法线：(-dzdx, -dzdy, 1)，再单位化
        normal_x = -dzdx
        normal_y = -dzdy
        normal_z = torch.ones_like(depth)

        normals = torch.cat([normal_x, normal_y, normal_z], dim=1)
        normals = F.normalize(normals, dim=1)

        return normals

    def _normal_consistency_loss(self, depth_pred, depth_gt):
        normal_pred = self.compute_normals(depth_pred)
        normal_gt   = self.compute_normals(depth_gt)

        dot_product = torch.sum(normal_pred * normal_gt, dim=1, keepdim=True)  # [B,1,H,W]
        loss_map = 1.0 - dot_product
        return loss_map.mean()

    def _gradient(self, pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    def _ssim_loss(self, x, y):
        '''
        from monodepth 
        https://github.com/mrharicot/monodepth/blob/b76bee4bd12610b482163871b7ff93e931cb5331/monodepth_model.py#L91
        '''
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1, 0)
        mu_y = F.avg_pool2d(y, 3, 1, 0)
        
        #(input, kernel, stride, padding)
        sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
        sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y
        
        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        SSIM = SSIM_n / SSIM_d
        
        return torch.clamp((1 - SSIM) / 2, 0, 1)
    
    def load_calib_ini(self, filepath):
        config = configparser.ConfigParser()
        config.read(filepath)
        
        cx = float(config['Intrinsics']['cx'])
        cy = float(config['Intrinsics']['cy'])
        fx = float(config['Intrinsics']['fx'])
        fy = float(config['Intrinsics']['fy'])

        mtx = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])
        
        dist_coeffs = np.array([
            float(config['Intrinsics'].get('k1', 0)),
            float(config['Intrinsics'].get('k2', 0)),
            float(config['Intrinsics'].get('p1', 0)),
            float(config['Intrinsics'].get('p2', 0)),
            float(config['Intrinsics'].get('k3', 0))
        ])

        return {'mtx': mtx, 'dist_coeffs': dist_coeffs}
    
    def load_camera_matrix(self, cam_path):
        
        np.seterr(divide='ignore',invalid='ignore')

        self.UW_calib = self.load_calib_ini(os.path.join(cam_path, 'UW_calib.ini'))
        self.I_calib = self.load_calib_ini(os.path.join(cam_path, 'I_calib.ini'))
        with open(os.path.join(cam_path, 'Rt_I_UW.pkl'),'rb') as f:
            self.I_UW_pose = pickle.load(f)
        with open(os.path.join(cam_path, 'Rt_UW_I.pkl'),'rb') as f:
            self.UW_I_pose = pickle.load(f)

        self.focal_length_x = self.UW_calib['mtx'][0,0]
        self.baseline = 10.0

        S_I_UW = self.I_calib['mtx']/self.UW_calib['mtx']
        self.scale_I_UW = (S_I_UW[0,0]+ S_I_UW[1,1])/2

        Rt_I_UW = self.I_UW_pose
        Rt_UW_I = self.UW_I_pose

        self.Rt_I_UW = torch.tensor(np.vstack((Rt_I_UW, np.array([[0, 0, 0, 1]])))).unsqueeze(0).float()
        self.Rt_UW_I = torch.tensor(np.vstack((Rt_UW_I, np.array([[0, 0, 0, 1]])))).unsqueeze(0).float()
        self.K_I  = torch.tensor(self.I_calib['mtx']).unsqueeze(0).float()
        self.K_UW = torch.tensor(self.UW_calib['mtx']).unsqueeze(0).float()

    def UWwarp2I_by_UW_depth(self, depth, warp_src):
        B, N, H, W = depth.size()
        mesh_grid = K.create_meshgrid(H, W ).cuda()
        UW_3d_pts = K.geometry.depth_to_3d_v2(depth, self.K_UW.squeeze(0).cuda(), normalize_points=True)
        UW_3d_pts_I = K.geometry.transform_points(self.Rt_UW_I.repeat(B, 1, 1, 1).cuda(), UW_3d_pts)
        UW_2d_pts_I = K.geometry.project_points(UW_3d_pts_I.reshape(B, H * W, 3), 
                                                self.K_I.repeat( H * W, 1, 1).cuda()).reshape(B, H, W, 2)
        I2UW_warp_grid = K.geometry.normalize_pixel_coordinates(UW_2d_pts_I, H, W)
        I2UW_flow = I2UW_warp_grid - mesh_grid
        UW2I_warp_grid = mesh_grid - I2UW_flow / self.scale_I_UW
        warped = F.grid_sample(warp_src, UW2I_warp_grid, align_corners=True)
        return warped

    def Iwarp2UW_by_I_depth(self, depth, warp_src):
        B, N, H, W = depth.size()
        mesh_grid = K.create_meshgrid(H, W ).cuda()
        I_3d_pts = K.geometry.depth_to_3d_v2(depth, self.K_I.squeeze(0).cuda(), normalize_points=True)
        I_3d_pts_UW = K.geometry.transform_points(self.Rt_I_UW.repeat(B, 1, 1, 1).cuda(), I_3d_pts)
        I_2d_pts_UW = K.geometry.project_points(I_3d_pts_UW.reshape(B, H * W, 3),
                                                self.K_UW.repeat( H * W, 1, 1).cuda()).reshape(B, H, W, 2)
        UW2I_warp_grid = K.geometry.normalize_pixel_coordinates(I_2d_pts_UW, H, W)
        UW2I_flow = UW2I_warp_grid-mesh_grid
        I2UW_warp_grid = mesh_grid-UW2I_flow*self.scale_I_UW
        warped = F.grid_sample(warp_src, I2UW_warp_grid, align_corners=True)
        return warped

    def _normalize_depth(self, depth):
        return depth/(depth.mean(3, True).mean(2, True)+ EPS)

    def init_midas(self, midas_mode, resize4midas):
        if self.structure_loss:
            self.midas_mode = midas_mode
            self.midas = torch.hub.load("intel-isl/MiDaS", self.midas_mode).cuda()
            self.resize4midas = resize4midas
            self.midas_transform = []
            if self.resize4midas:
                if self.midas_mode[-5:] =='small':
                    self.midas_transform += [torchvision.transforms.Resize(256, torchvision.transforms.InterpolationMode.BICUBIC),]
                else:
                    self.midas_transform += [torchvision.transforms.Resize(384, torchvision.transforms.InterpolationMode.BICUBIC),]
            
            if self.midas_mode[:3] =='DPT':
                self.midas_transform += [torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
            else:
                self.midas_transform += [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

            self.midas_transform = torchvision.transforms.Compose(self.midas_transform)

    def load_dataset(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def init_loss(self, regression_loss, structure_loss, smooth_loss, consistency_loss):
        self.regression_loss = regression_loss
        self.structure_loss = structure_loss
        self.smooth_loss = smooth_loss
        self.consistency_loss = consistency_loss

    def init_loss_weights(self, regression_loss_weight, smooth_loss_weight, structure_loss_weight, consistency_loss_weight):

        self.weights = dict()

        for index in range(5):
            self.weights[f'regression_loss_{index}'] = regression_loss_weight

            if self.structure_loss:
                self.weights[f'structure_ssim_loss_{index}'] = structure_loss_weight

            if self.smooth_loss:
                self.weights[f'smooth_loss_{index}'] = smooth_loss_weight

            if self.consistency_loss:
                self.weights[f'consistency_loss_{index}'] = consistency_loss_weight

    def compute_depth_metrics(self, predict, ground_truth):
        '''
        borrow by https://github.com/dusty-nv/pytorch-depth/blob/master/metrics.py
        '''
        valid_mask = ground_truth>0
        predict = predict[valid_mask]
        ground_truth = ground_truth[valid_mask]

        abs_diff = (predict - ground_truth).abs()
        mse = torch.pow(abs_diff, 2).mean()
        rmse = torch.sqrt(mse)
        mae = abs_diff.mean()
        log_diff = torch.log10(predict) - torch.log10(ground_truth)
        lg10 = log_diff.abs().mean()
        rmse_log = torch.sqrt(torch.pow(log_diff, 2).mean())
        absrel = float((abs_diff / ground_truth).mean())
        sqrel = float((torch.pow(abs_diff, 2) / ground_truth).mean())

        maxRatio = torch.max(predict / ground_truth, ground_truth / predict)
        delta1 = (maxRatio < 1.25).float().mean()
        delta2 = (maxRatio < 1.25 ** 2).float().mean()
        delta3 = (maxRatio < 1.25 ** 3).float().mean()
        return mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log

    def _adjust_learning_rate(self, ):
        lr = self.lr / (1.0 + self.lr_decay * self.n_iter)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def fit(self, epochs):
        for epoch in tqdm(range(epochs)):
            self._epoch()
        self._save_model_cpt()

    def _visualize_depths(self, depth_tag, depth, save_path):
        """
        Visualize depth: local min-max normalization + jet colormap (consistent with depth_to_color)
        """
        B, C, H, W = depth.size()
        depth = depth.detach()
        depth_tag = depth_tag.split('/')

        for i in range(B):
            depth_i = depth[i].squeeze(0).cpu().numpy() * self.depth_scale

            # Local normalization
            dmin, dmax = np.min(depth_i), np.max(depth_i)
            if dmax - dmin > 1e-5:
                norm_depth = (depth_i - dmin) / (dmax - dmin)
            else:
                norm_depth = np.zeros_like(depth_i)

            depth_figure = plt.figure(figsize=(6, 4))
            plt.imshow(norm_depth, cmap='jet')  # 和 depth_to_color 一致
            #plt.colorbar()
            plt.axis('off')

            plt.savefig(
                os.path.join(save_path, f'{depth_tag[-1]}_{i}.png'),
                dpi=100
            )

            self.tb_writer.add_figure(
                f'{depth_tag[0]}/{depth_tag[-1]}_{i}',
                depth_figure,
                self.n_iter
            )

            plt.clf()

    def _visualize_structure_depths(self, depth_tag, depth, save_path):
        B, C, H, W = depth.size()
        depth = depth.detach()
        depth_tag = depth_tag.split('/')
        for i in range(B):
            depth_i = depth[i].squeeze(0).cpu().numpy()
            depth_figure = plt.figure(figsize=(6, 4))
            plt.imshow(depth_i, cmap='plasma_r', vmin=0, vmax=1.)
            plt.colorbar()
            plt.axis('off')
            plt.savefig(os.path.join(save_path, f'{depth_tag[-1]}_{i}.png'), dpi=100)
            self.tb_writer.add_figure(f'{depth_tag[0]}/{depth_tag[-1]}_{i}', depth_figure, self.n_iter)
            plt.clf() 

    def init_log(self, tb_writer, img_dir, cpt_dir):
        self.tb_writer = tb_writer
        self.img_dir = img_dir
        self.cpt_dir = cpt_dir
    
    def init_other_settings(self, depth_scale, mode, struct_knowledge_src):
        self.mode = mode
        self.depth_scale = depth_scale
        self.struct_knowledge_src = struct_knowledge_src
    
    def __init__(self, ):
        super().__init__()
        self.uw_best_mae_epoch = False
        self.uw_best_mae = self.DEFAULT_MAX_MAE
    
    def init_optim(self,):
        raise NotImplementedError

    def init_val_data(self,):
        raise NotImplementedError

    def _compute_losses(self,):
        raise NotImplementedError

    def _compute_val_metrics(self,):
        raise NotImplementedError

    def _epoch_fit(self, ):
        raise NotImplementedError

    def _epoch_val(self,):
        raise NotImplementedError

    def _epoch(self,):
        raise NotImplementedError

    def _visualize(self,):
        raise NotImplementedError

class TFT3D_SingleSplitModelTrainer(BasicTrainer):
    def __init__(self, uw_model):
        super().__init__()
        self.uw_model  = uw_model.cuda() 

        self.n_iter = 0
        self.epoch = 0

    def init_optim(self, lr, betas, lr_decay, use_radam, use_lr_decay,
                   fixed_img_encoder ):
        print('init optimizer....')
        self.lr = lr 
        self.betas = betas
        self.use_radam = use_radam
        self.lr_decay = lr_decay
        self.use_lr_decay = use_lr_decay

        self.fixed_img_encoder = fixed_img_encoder

        update_models = []
        if fixed_img_encoder:
            update_models.append(self.uw_model.D)
        else:
            update_models.append(self.uw_model)

        update_models = nn.ModuleList(update_models)

        if self.use_radam:
            self.opt = torch.optim.RAdam(update_models.parameters(), lr=self.lr, betas=self.betas)
        else:
            self.opt = torch.optim.Adam(update_models.parameters(), lr=self.lr, betas=self.betas)

    def init_val_data(self,):
        gt_depth_uw, rgb_uw, ndepth_uw, conf_uw = next(iter(self.val_dataset))
        gt_depth_uw = gt_depth_uw.cuda()
        rgb_uw = rgb_uw.cuda()
        if 'RGBD' in self.mode:
            ndepth_uw = ndepth_uw.cuda()
        else:
            ndepth_uw = None
        conf_uw = conf_uw.cuda()

        tof_depth = None
        self.val_conf_uw = conf_uw.cpu()  
        self.val_rgb_uw = rgb_uw.cpu()
        self.val_input_depth_uw = None
        input_depth_uw = None

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            input_depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            self.val_input_depth_uw = input_depth_uw.cpu()

        save_path = os.path.join(self.img_dir, 'ground_truth')
        os.makedirs(save_path, exist_ok=True)

        if tof_depth is not None:
            tof_depth /= self.depth_scale
            self.tb_writer.add_images(f'ground_truth/tof_depth', tof_depth, self.n_iter)
            torchvision.utils.save_image(tof_depth, os.path.join(save_path, f'tof_depth.png')) 
            self._visualize_depths('ground_truth/tof_depth', tof_depth, save_path)

        if self.val_input_depth_uw is not None:
            self.tb_writer.add_images(f'input/depth_uw', self.val_input_depth_uw, self.n_iter)
            torchvision.utils.save_image(self.val_input_depth_uw, os.path.join(save_path, f'input_depth_uw.png')) 
            self._visualize_depths('input/input_depth_uw', self.val_input_depth_uw, save_path)

        self.tb_writer.add_images(f'ground_truth/rgb_uw', rgb_uw, self.n_iter)
        torchvision.utils.save_image(rgb_uw, os.path.join(save_path, f'rgb_uw.png'))
        gt_depth_uw /= self.depth_scale
        self.tb_writer.add_images(f'ground_truth/gt_depth_uw', gt_depth_uw, self.n_iter)
        torchvision.utils.save_image(gt_depth_uw, os.path.join(save_path, f'gt_depth_uw.png')) 
        self._visualize_depths('gt_depth_uw/gt_depth_uw', gt_depth_uw, save_path)

        gt_depth_uw_conf = gt_depth_uw * conf_uw
        self.tb_writer.add_images(f'ground_truth/gt_depth_uw_conf', gt_depth_uw_conf, self.n_iter)
        torchvision.utils.save_image(gt_depth_uw_conf, os.path.join(save_path, f'gt_depth_uw_conf.png')) 
        self._visualize_depths('gt_depth_uw/gt_depth_uw_conf', gt_depth_uw_conf, save_path)

        if self.structure_loss:
            monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
            monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_uw = 1 - monodepth_uw
            if self.resize4midas:
                monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                mode="bicubic")
            torchvision.utils.save_image(monodepth_uw, os.path.join(save_path, f'monodepth_uw.png')) 
            self._visualize_depths('gt_depth_uw/monodepth_uw', monodepth_uw, save_path)

    def _compute_losses(self, depths_pred_uw, rgb_uw, monodepth_uw ,gt_depth_uw, conf_uw):
        losses = dict()

        for index in range(5):
            depth_i_pred_uw = depths_pred_uw[index]
            if self.regression_loss == 'l1':
                losses[f'regression_loss_{index}'] = F.l1_loss(depth_i_pred_uw*conf_uw, gt_depth_uw*conf_uw/ self.depth_scale)
            elif self.regression_loss == 'l2':
                losses[f'regression_loss_{index}'] = F.mse_loss(depth_i_pred_uw*conf_uw, gt_depth_uw*conf_uw/ self.depth_scale)
            elif self.regression_loss == 'smooth_l1':
                losses[f'regression_loss_{index}'] = F.smooth_l1_loss(depth_i_pred_uw*conf_uw, gt_depth_uw*conf_uw/ self.depth_scale)
            if self.smooth_loss:
                losses[f'smooth_loss_{index}'] = self._smooth_loss(depth_i_pred_uw, rgb_uw)
            if self.structure_loss:
                losses[f'structure_ssim_loss_{index}'] = self._ssim_loss(self._normalize_depth(depth_i_pred_uw), self._normalize_depth(monodepth_uw)).mean()
            if self.consistency_loss:
                losses[f'consistency_loss_{index}'] = self._normal_consistency_loss(depth_i_pred_uw*conf_uw, gt_depth_uw*conf_uw/ self.depth_scale)

        return losses

    def _compute_val_metrics(self, depth_pred_uw, gt_depth_uw, conf_uw, tof_uw_mask=None):

        metrics = dict()
    
        mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
            gt_depth_uw/10 * conf_uw, depth_pred_uw * self.depth_scale /10 *conf_uw)

        metrics['depth_uw/mse'] = mse
        metrics['depth_uw/rmse'] = rmse
        metrics['depth_uw/mae'] = mae
        metrics['depth_uw/lg10'] = lg10
        metrics['depth_uw/absrel'] = absrel
        metrics['depth_uw/delta1'] = delta1
        metrics['depth_uw/delta2'] = delta2
        metrics['depth_uw/delta3'] = delta3
        metrics['depth_uw/sqrel'] = sqrel
        metrics['depth_uw/rmse_log'] = rmse_log

        # l1
        metrics['depth_uw/l1'] = F.l1_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale*conf_uw)
        # l2
        metrics['depth_uw/l2'] = F.mse_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale * conf_uw)
        # SSIM
        metrics['depth_uw/ssim'] = K.metrics.ssim(gt_depth_uw / self.depth_scale  * conf_uw, depth_pred_uw * conf_uw, 11).mean()
        # PSNR
        metrics['depth_uw/psnr'] = K.metrics.psnr(gt_depth_uw / self.depth_scale * conf_uw, depth_pred_uw * conf_uw, 1)

        if tof_uw_mask is not None:
            mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                gt_depth_uw/10 * conf_uw*tof_uw_mask, depth_pred_uw * self.depth_scale /10 *conf_uw*tof_uw_mask)

            metrics['our_of_tof/depth_uw/mse'] = mse
            metrics['our_of_tof/depth_uw/rmse'] = rmse
            metrics['our_of_tof/depth_uw/mae'] = mae
            metrics['our_of_tof/depth_uw/lg10'] = lg10
            metrics['our_of_tof/depth_uw/absrel'] = absrel
            metrics['our_of_tof/depth_uw/delta1'] = delta1
            metrics['our_of_tof/depth_uw/delta2'] = delta2
            metrics['our_of_tof/depth_uw/delta3'] = delta3
            metrics['our_of_tof/depth_uw/sqrel'] = sqrel
            metrics['our_of_tof/depth_uw/rmse_log'] = rmse_log

            overlay_uw_mask = tof_uw_mask==False

            mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                gt_depth_uw/10 * conf_uw*overlay_uw_mask, depth_pred_uw * self.depth_scale /10 *conf_uw*overlay_uw_mask)

            metrics['overlap_tof/depth_uw/mse'] = mse
            metrics['overlap_tof/depth_uw/rmse'] = rmse
            metrics['overlap_tof/depth_uw/mae'] = mae
            metrics['overlap_tof/depth_uw/lg10'] = lg10
            metrics['overlap_tof/depth_uw/absrel'] = absrel
            metrics['overlap_tof/depth_uw/delta1'] = delta1
            metrics['overlap_tof/depth_uw/delta2'] = delta2
            metrics['overlap_tof/depth_uw/delta3'] = delta3
            metrics['overlap_tof/depth_uw/sqrel'] = sqrel
            metrics['overlap_tof/depth_uw/rmse_log'] = rmse_log
        return metrics

    def _warp_train_data(self, gt_depth_uw, rgb_uw, ndepth_uw, conf_uw):
        uw_model_input = {'x_img':rgb_uw}

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            uw_model_input['x_depth']= depth_uw

        train_loss_data = {'rgb_uw':rgb_uw,
                      'monodepth_uw': None,
                      'gt_depth_uw' : gt_depth_uw,
                      'conf_uw' : conf_uw}

        if self.structure_loss:
            monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
            monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_uw = 1 - monodepth_uw
            if self.resize4midas:
                monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                mode="bicubic")
            train_loss_data['monodepth_uw'] = monodepth_uw

        return uw_model_input, train_loss_data
    
    def _epoch_fit(self, ):
        epoch_pbar = tqdm(self.train_dataset, desc=f"Epoch {self.epoch}", leave=True)
        for gt_depth_uw, rgb_uw, ndepth_uw, conf_uw in epoch_pbar:

            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()
            conf_uw = conf_uw.cuda()

            if 'RGBD' in self.mode:
                ndepth_uw = ndepth_uw.cuda()
            else:
                ndepth_uw = None

            with torch.no_grad():
                uw_model_input, train_loss_data = \
                    self._warp_train_data(gt_depth_uw, rgb_uw, ndepth_uw, conf_uw)
                del gt_depth_uw, ndepth_uw

            depths_pred = self.uw_model(**uw_model_input)

            train_loss_data['depths_pred_uw'] = depths_pred

            losses = self._compute_losses(**train_loss_data)

            total_loss = 0
            for loss_name in losses.keys():
                loss = losses[loss_name]
                weight = self.weights[loss_name]
                self.tb_writer.add_scalar('train_loss/{}'.format(loss_name), loss, self.n_iter)
                weighted_loss = loss * weight
                self.tb_writer.add_scalar('weighted_train_loss/{}'.format(loss_name), weighted_loss, self.n_iter)
                total_loss += weighted_loss

            self.opt.zero_grad()
            if self.use_lr_decay:
                self._adjust_learning_rate()
            total_loss.backward()
            self.opt.step()

            self.tb_writer.add_scalar('train_loss/total_loss', total_loss, self.n_iter)

            self.n_iter+=1

            # 实时更新进度条上的 loss
            epoch_pbar.set_postfix({'loss': total_loss.item()})
    
    def _warp_val_data(self, gt_depth_uw, rgb_uw, ndepth_uw, conf_uw):
        
        uw_model_input = {'x_img':rgb_uw}
            
        val_loss_data = {'rgb_uw':rgb_uw,
                      'monodepth_uw': None,
                      'gt_depth_uw' : gt_depth_uw,
                      'conf_uw' : conf_uw}
        
        val_metric_data = {'gt_depth_uw': gt_depth_uw,
                            'conf_uw': conf_uw,
                            'tof_uw_mask': None,}

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            uw_model_input['x_depth'] = depth_uw
            tof_uw_mask = depth_uw==0
            val_metric_data['tof_uw_mask'] = tof_uw_mask

        if self.structure_loss:
            monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
            monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_uw = 1 - monodepth_uw
            if self.resize4midas:
                monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                mode="bicubic")
            val_loss_data['monodepth_uw'] = monodepth_uw

        return uw_model_input, val_loss_data, val_metric_data

    def _epoch_val(self,):
        print('epoch testing...')
        val_pbar = tqdm(iter(self.val_dataset))
        i=0
        print('computing quantative evaluations....')
        for gt_depth_uw, rgb_uw, ndepth_uw, conf_uw in val_pbar:
            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()

            if 'RGBD' in self.mode:
                ndepth_uw = ndepth_uw.cuda()
            else:
                ndepth_uw = None

            conf_uw = conf_uw.cuda()

            uw_model_input, val_loss_data, val_metric_data = \
                self._warp_val_data(gt_depth_uw, rgb_uw, ndepth_uw, conf_uw)

            depths_pred = self.uw_model(**uw_model_input)

            val_loss_data['depths_pred_uw'] = depths_pred
            val_metric_data['depth_pred_uw'] = depths_pred[-1]

            if i==0:
                total_losses = self._compute_losses(**val_loss_data)
    
                total_metrics = self._compute_val_metrics(**val_metric_data)

            else:
                losses = self._compute_losses(**val_loss_data)
                metrics = self._compute_val_metrics(**val_metric_data)

                for loss_name in losses.keys():
                    total_losses[loss_name] += losses[loss_name]

                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]
            i+=1

        total_loss = 0
        weighted_total_loss = 0
        for loss_name in total_losses.keys():
            loss = total_losses[loss_name]/i
            self.tb_writer.add_scalar('val_loss/{}'.format(loss_name), loss, self.n_iter)
            total_loss+=loss 
            weighted_loss = loss * self.weights[loss_name]
            self.tb_writer.add_scalar('weighted_val_loss/{}'.format(loss_name), weighted_loss, self.n_iter)
            weighted_total_loss += weighted_loss

        self.tb_writer.add_scalar('val_loss/total_loss', total_loss, self.n_iter)
        self.tb_writer.add_scalar('val_loss/weighted_total_loss', weighted_total_loss, self.n_iter)

        for metric_name in metrics.keys():
            metric = total_metrics[metric_name]/i
            self.tb_writer.add_scalar('metrics/{}'.format(metric_name), metric, self.n_iter)

            if metric_name == 'depth_uw/mae':
                if metric < self.uw_best_mae:
                    self.uw_best_mae_epoch = True
                    self.uw_best_mae = metric
                else:
                    self.uw_best_mae_epoch = False 

    def _epoch(self,):
        if self.fixed_img_encoder:
            self.uw_model.E_img.train(False)
            self.uw_model.D.train(True)
        else:
            self.uw_model.train(True)
        self._epoch_fit()
        self.uw_model.train(False)
        with torch.no_grad():
            self._visualize()
            self._epoch_val()
            if self.uw_best_mae_epoch:
                self._save_model_cpt()
        self.epoch+=1

    def _visualize(self,):
        
        if self.val_input_depth_uw is not None:
            depths_pred_uw = self.uw_model(self.val_rgb_uw.cuda(), self.val_input_depth_uw.cuda())
        else:
            depths_pred_uw = self.uw_model(self.val_rgb_uw.cuda())

        save_path = os.path.join(self.img_dir, '{}'.format(self.n_iter))
        os.makedirs(save_path, exist_ok=True)

        for index in trange(5):
            depth_i_pred_uw = depths_pred_uw[index]

            self.tb_writer.add_images(f'test/depth_uw_{index}', depth_i_pred_uw, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_uw, os.path.join(save_path, f'depth_pred_uw_{index}.png'))
            self._visualize_depths(f'val_depth_{index}/depth_pred_uw_{index}', depth_i_pred_uw, save_path) 
            
            depth_i_pred_uw_conf = depth_i_pred_uw * self.val_conf_uw.cuda()
            self.tb_writer.add_images(f'test/depth_uw_{index}_conf', depth_i_pred_uw_conf, self.n_iter)
            torchvision.utils.save_image(depth_i_pred_uw_conf, os.path.join(save_path, f'depth_pred_uw_{index}_conf.png'))
            self._visualize_depths(f'val_depth_{index}/depth_pred_uw_{index}_conf', depth_i_pred_uw_conf, save_path) 

    def _save_model_cpt(self,):
        save_path = os.path.join(self.cpt_dir, 'UW_E{}_iter_{}.cpt'.format(self.epoch, self.n_iter))
        print('saving uw model cpt @ {}'.format(save_path))
        print('UW MAE(cm): {}'.format(self.uw_best_mae))
        torch.save(self.uw_model.state_dict(), save_path)
