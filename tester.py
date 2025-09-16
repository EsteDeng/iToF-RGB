import os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import configparser
import cv2
import pandas as pd
import kornia as K
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from scipy.io import savemat
from kornia.geometry.depth import depth_to_3d
class BasicTester(nn.Module):
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

    def create_meshgrid(self,H, W, normalized=True, device="cpu"):
        ys, xs = torch.meshgrid(
            torch.linspace(0, H - 1, H, device=device),
            torch.linspace(0, W - 1, W, device=device),
            indexing="ij"
        )
        grid = torch.stack([xs, ys], dim=-1)  # [H,W,2]
        if normalized:
            xs = (xs / (W - 1)) * 2 - 1
            ys = (ys / (H - 1)) * 2 - 1
            grid = torch.stack([xs, ys], dim=-1)
        return grid.unsqueeze(0)  # [1,H,W,2]

    def _as_batched_K(self, K_raw, B, device, dtype=torch.float32):
        K_ = torch.as_tensor(K_raw, device=device, dtype=dtype)
        K_ = K_.squeeze()                        # remove redundant dimensions
        K_ = K_.view(3, 3)                       # force reshape to 3x3
        K_ = K_.unsqueeze(0).expand(B, -1, -1)   # expand to [B, 3, 3]
        return K_

    def _as_batched_RT(self,RT_raw, B, device, dtype=torch.float32):
        RT = torch.as_tensor(RT_raw, device=device, dtype=dtype).squeeze()
        # 支持 3x4 或 4x4
        if RT.dim() != 2 or (RT.shape[-2:] not in [(3, 4), (4, 4)]):
            raise ValueError(f"Rt_UW_I must be 3x4 or 4x4, got {tuple(RT.shape)}")
        return RT.unsqueeze(0).expand(B, -1, -1)   # [B,3x4] or [B,4x4]

    def UWwarp2I_by_UW_depth(self, depth, warp_src):
        B, _, H, W = depth.shape
        device = depth.device
        dtype  = depth.dtype

        # mesh grid ([-1,1] normalized)
        try:
            from kornia.utils.grid import create_meshgrid
            mesh_grid = create_meshgrid(H, W, normalized=True, device=device).expand(B, -1, -1, -1)  # [B,H,W,2]
        except Exception:
            ys, xs = torch.meshgrid(
                torch.linspace(0, H - 1, H, device=device, dtype=dtype),
                torch.linspace(0, W - 1, W, device=device, dtype=dtype),
                indexing="ij"
            )
            xs = (xs / (W - 1)) * 2 - 1
            ys = (ys / (H - 1)) * 2 - 1
            mesh_grid = torch.stack([xs, ys], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        K_uw = self._as_batched_K(self.K_UW, B, device, dtype)
        K_i  = self._as_batched_K(self.K_I,  B, device, dtype)

        # UW depth -> 3D points in UW camera coordinates
        # If you prefer using depth_to_3d_v2: from kornia.geometry.depth import depth_to_3d_v2 as depth_to_3d
        UW_3d = depth_to_3d(depth, K_uw)                 # [B,3,H,W]

        # UW -> I extrinsic matrix
        Rt = self._as_batched_RT(self.Rt_UW_I, B, device, dtype)

        # transform to I camera coordinates
        pts3d = UW_3d.permute(0, 2, 3, 1).reshape(B, -1, 3)  # [B,HW,3]
        pts3d_I = K.geometry.transform_points(Rt, pts3d)     # [B,HW,3]

        # project to I image pixels
        uv_I = K.geometry.project_points(pts3d_I, K_i)       # [B,HW,2]
        uv_I = uv_I.view(B, H, W, 2)                         # [B,H,W,2]
        grid_I = K.geometry.normalize_pixel_coordinates(uv_I, H, W)  # [-1,1]

        # compute flow & inverse sampling grid
        I2UW_flow  = grid_I - mesh_grid
        UW2I_grid  = mesh_grid - I2UW_flow / self.scale_I_UW

        warped = F.grid_sample(warp_src, UW2I_grid, align_corners=True)
        return warped

    def _create_meshgrid(self,H, W, device, dtype):
        try:
            from kornia.utils.grid import create_meshgrid
            mg = create_meshgrid(H, W, normalized=True, device=device, dtype=dtype)  # [1,H,W,2]
        except Exception:
            ys, xs = torch.meshgrid(
                torch.linspace(0, H - 1, H, device=device, dtype=dtype),
                torch.linspace(0, W - 1, W, device=device, dtype=dtype),
                indexing="ij"
            )
            xs = (xs / (W - 1)) * 2 - 1
            ys = (ys / (H - 1)) * 2 - 1
            mg = torch.stack([xs, ys], dim=-1).unsqueeze(0)  # [1,H,W,2]
        return mg
    
    def Iwarp2UW_by_I_depth(self, depth, warp_src):
        """
        depth:    [B,1,H,W]  Depth from I camera, in the same unit as intrinsics
        warp_src: [B,C,H,W]  Image to be warped onto the UW plane
        """
        device, dtype = depth.device, depth.dtype
        B, _, H, W = depth.shape

        # 1) base mesh grid ([-1,1] normalized)
        mesh_grid = self._create_meshgrid(H, W, device, dtype).expand(B, -1, -1, -1)  # [B,H,W,2]

        # 2) prepare batched camera intrinsics
        K_i  = self._as_batched_K(self.K_I,  B, device, dtype)    # [B,3,3]
        K_uw = self._as_batched_K(self.K_UW, B, device, dtype)    # [B,3,3]

        # 3) I depth -> 3D points in I camera coordinates
        I_3d = depth_to_3d(depth, K_i)                       # [B,3,H,W]
        pts3d_I = I_3d.permute(0, 2, 3, 1).reshape(B, -1, 3) # [B,HW,3]

        # 4) I -> UW extrinsic matrix
        Rt_I_UW = self._as_batched_RT(self.Rt_I_UW, B, device, dtype)

        # 5) transform 3D points into UW camera coordinates
        pts3d_UW = K.geometry.transform_points(Rt_I_UW, pts3d_I)   # [B,HW,3]

        # 6) project to UW image plane
        uv_UW = K.geometry.project_points(pts3d_UW, K_uw)         # [B,HW,2]
        uv_UW = uv_UW.view(B, H, W, 2)

        # 7) normalize to [-1,1] to get the UW sampling grid
        UW_warp_grid = K.geometry.normalize_pixel_coordinates(uv_UW, H, W)  # [B,H,W,2]

        # 8) compute flow relative to the original mesh grid, then build I→UW sampling grid using your scale factor
        UW2I_flow      = UW_warp_grid - mesh_grid         # keeping the original naming convention
        I2UW_warp_grid = mesh_grid - UW2I_flow * self.scale_I_UW

        # 9) sampling
        warped = F.grid_sample(warp_src, I2UW_warp_grid, align_corners=True)
        return warped


    def compute_depth_metrics(self, predict, ground_truth):
        '''
        borrow by https://github.com/dusty-nv/pytorch-depth/blob/master/metrics.py
        '''
        valid_mask = ground_truth>0
        predict = predict[valid_mask]
        ground_truth = ground_truth[valid_mask]

        abs_diff = (predict - ground_truth).abs()
        mse = torch.pow(abs_diff, 2).mean()
        rmse = torch.sqrt(mse).cpu().item()
        mae = abs_diff.mean().cpu().item()
        log_diff = torch.log10(predict) - torch.log10(ground_truth)
        lg10 = log_diff.abs().mean().cpu().item()
        rmse_log = torch.sqrt(torch.pow(log_diff, 2).mean()).cpu().item()
        absrel = float((abs_diff / ground_truth).mean())
        sqrel = float((torch.pow(abs_diff, 2) / ground_truth).mean())

        maxRatio = torch.max(predict / ground_truth, ground_truth / predict)
        delta1 = (maxRatio < 1.25).float().mean().cpu().item()
        delta2 = (maxRatio < 1.25 ** 2).float().mean().cpu().item()
        delta3 = (maxRatio < 1.25 ** 3).float().mean().cpu().item()
        return mse.cpu().item(), rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log

    def _compute_test_metrics(self, gt_depth_uw, 
                                    conf_uw, tof_uw_mask, 
                                    depth_pred_uw=None):

        metrics = dict()
        #metrics
        #UW depth
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
        metrics['depth_uw/l1'] = F.l1_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale*conf_uw).cpu().item()
        # l2
        metrics['depth_uw/l2'] = F.mse_loss(gt_depth_uw * conf_uw, depth_pred_uw * self.depth_scale * conf_uw).cpu().item()
        # SSIM
        metrics['depth_uw/ssim'] = K.metrics.ssim(gt_depth_uw / self.depth_scale  * conf_uw, depth_pred_uw * conf_uw, 11).mean().cpu().item()
        # PSNR
        metrics['depth_uw/psnr'] = K.metrics.psnr(gt_depth_uw / self.depth_scale * conf_uw, depth_pred_uw * conf_uw, 1).cpu().item()

        if tof_uw_mask is not None:
            mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, sqrel, rmse_log = self.compute_depth_metrics(
                gt_depth_uw/10 * conf_uw*tof_uw_mask, depth_pred_uw * self.depth_scale /10 *conf_uw*tof_uw_mask)

            metrics['out_of_tof/depth_uw/mse'] = mse
            metrics['out_of_tof/depth_uw/rmse'] = rmse
            metrics['out_of_tof/depth_uw/mae'] = mae
            metrics['out_of_tof/depth_uw/lg10'] = lg10
            metrics['out_of_tof/depth_uw/absrel'] = absrel
            metrics['out_of_tof/depth_uw/delta1'] = delta1
            metrics['out_of_tof/depth_uw/delta2'] = delta2
            metrics['out_of_tof/depth_uw/delta3'] = delta3
            metrics['out_of_tof/depth_uw/sqrel'] = sqrel
            metrics['out_of_tof/depth_uw/rmse_log'] = rmse_log

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

class SingleModelTester_tft(BasicTester):

    def __init__(self, cam_path, depth_scale, model, mode, midas_mode='MiDaS', resize4midas=True, line_pos=(150, 200), save_dir='test_visuals'):
        super().__init__()
        self.cam_path = cam_path
        self.load_camera_matrix(cam_path)
        self.depth_scale = depth_scale
        self.midas_mode = midas_mode
        self.model  = model.cuda() 

        self.mode = mode
        self.line_pos = line_pos
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.init_midas(midas_mode, resize4midas)

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

    
    def get_relative_depth(self, rgb_uw):
        with torch.no_grad():
            midas_input = self.midas_transform(rgb_uw)
            midas_depth = self.midas(midas_input)
            midas_depth = midas_depth.unsqueeze(1)
            midas_depth = midas_depth / midas_depth.view(midas_depth.size(0), -1).max(1)[0].view(-1, 1, 1, 1)
            midas_depth = 1 - midas_depth
            if self.resize4midas:
                midas_depth = F.interpolate(midas_depth, size=rgb_uw.shape[2:], mode='bicubic', align_corners=True)
            return midas_depth

    def depth_to_color(self, depth_tensor):
        depth = depth_tensor.cpu().numpy()
        dmin, dmax = np.min(depth), np.max(depth)
        print(f"Depth range: {dmin} to {dmax}")
        dmin, dmax = np.min(depth), np.max(depth)
        if dmax - dmin > 1e-5:
            norm_depth = (depth - dmin) / (dmax - dmin)
        else:
            norm_depth = np.zeros_like(depth)
        cmap = plt.get_cmap('jet')
        colored = cmap(norm_depth)[:, :, :3] 
        colored_img = (colored * 255).astype(np.uint8)
        pil_img = Image.fromarray(colored_img)
        return pil_img

    def draw_lines(self, pil_img, line_pos, color_h=(0, 0, 255), color_v=(0, 255, 0)):
        """
        Draw horizontal and vertical lines on a PIL image.
        line_pos: (y, x)
        color_h: color of the horizontal line in BGR format, default is blue (0,0,255)
        color_v: color of the vertical line in BGR format, default is green (0,255,0)
        """
        draw = ImageDraw.Draw(pil_img)
        y, x = line_pos
        w, h = pil_img.size
        # draw horizontal line
        draw.line([(0, y), (w, y)], fill=color_h, width=8)
        # draw vertical line
        draw.line([(x, 0), (x, h)], fill=color_v, width=8)
        return pil_img

    def plot_line_depths(self, depths_dict, line_pos, save_path):
        """
        depths_dict: dict, keys are names and values are 2D tensors (H, W)
        line_pos: (y, x)
        """

        y, x = line_pos

        colors = ['#FFA500', '#FF00FF', '#FF0000', '#000000']  # yellow, magenta, red, black

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios':[1,1]})

        # Horizontal line depth (top plot)
        for i, (name, depth) in enumerate(depths_dict.items()):
            line_data = depth.cpu().numpy()[y, :]
            z = 1 if i == 3 else (i + 2)
            axs[0].plot(line_data, color=colors[i], zorder=z, linewidth=3.3)
        axs[0].grid(True)

        # Remove axis labels
        axs[0].set_xlabel('')
        axs[0].set_ylabel('')

        # Hide tick labels (but keep tick marks)
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])

        axs[0].yaxis.tick_right()
        axs[0].yaxis.set_label_position("right")
        for spine in axs[0].spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(4.5)

        # Vertical line depth (bottom plot)
        for i, (name, depth) in enumerate(depths_dict.items()):
            line_data = depth.cpu().numpy()[:, x]
            z = 1 if i == 3 else (i + 2)
            axs[1].plot(line_data, color=colors[i], zorder=z, linewidth=3.3)
        axs[1].grid(True)

        axs[1].set_xlabel('')
        axs[1].set_ylabel('')

        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])

        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")
        for spine in axs[1].spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(4.5)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _warp_test_data(self, gt_depth_uw, rgb_uw, ndepth_uw, conf_uw):
        model_input = {'x_img':rgb_uw}
        test_metric_data = {'gt_depth_uw': gt_depth_uw,
                            'conf_uw': conf_uw,
                            'tof_uw_mask': None}

        if ndepth_uw is not None:
            tof_depth = self.UWwarp2I_by_UW_depth(gt_depth_uw, ndepth_uw)
            depth_uw = self.Iwarp2UW_by_I_depth(tof_depth, tof_depth)/self.depth_scale
            model_input['x_depth']= depth_uw
            
            tof_uw_mask = depth_uw==0
            test_metric_data['tof_uw_mask'] = tof_uw_mask


        return model_input, test_metric_data ,tof_depth
    def plot_line_errors(self, depths_tensors, line_pos, save_dir, prefix=""):

        gt = depths_tensors['Ground Truth depth'].detach().cpu().numpy()
        curves = {k: v.detach().cpu().numpy() for k, v in depths_tensors.items() if k in ['relative depth', 'predicted depth']}

        x_pos, y_pos = line_pos

        colors = {
            'relative depth': '#FF00FF',
            'predicted depth': '#FF0000',
        }

        gt_h = gt[y_pos, :]
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, v in curves.items():
            err = np.abs(v[y_pos, :] - gt_h) / (gt_h + 1e-8) * 100
            ax.plot(err, color=colors.get(k, 'black'), linewidth=3)

        ax.axhline(5, color='gray', linestyle='--', linewidth=2) 

        ax.grid(True, linestyle='--', linewidth=0.8) 

        ax.set_xticklabels([]) 
        ax.tick_params(axis='y', labelsize=16)

        for spine in ax.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(2.5)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{prefix}line_error_horizontal.png")
        fig.savefig(save_path)
        plt.close(fig)

        gt_v = gt[:, x_pos]
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, v in curves.items():
            err = np.abs(v[:, x_pos] - gt_v) / (gt_v + 1e-8) * 100
            ax.plot(err, color=colors.get(k, 'black'), linewidth=3)

        ax.axhline(5, color='gray', linestyle='--', linewidth=2)
        ax.grid(True, linestyle='--', linewidth=0.8)

        ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=16)

        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2.5)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{prefix}line_error_vertical.png")
        fig.savefig(save_path)
        plt.close(fig)

        print(f"Saved: {save_path} with spine colored borders.")
    
    def _compute_psnr(self, pred_tensor, gt_tensor, max_val=None):
        """
        pred_tensor, gt_tensor: [H, W] (same scale and same unit)
        Compute PSNR only on valid pixels where gt > 0; 
        if max_val=None, automatically use the maximum value of valid GT pixels.
        Return value is in dB.
        """
        pred = pred_tensor.detach().float().cpu().numpy()
        gt   = gt_tensor.detach().float().cpu().numpy()

        valid = gt > 0
        if not np.any(valid):
            return float('nan')

        diff = pred[valid] - gt[valid]
        mse = float(np.mean(diff * diff))
        if mse == 0.0:
            return float('inf')

        if max_val is None:
            max_val = float(np.max(gt[valid]))
            if max_val <= 0:
                return float('nan')

        psnr = 10.0 * np.log10((max_val ** 2) / mse)
        return psnr

    @torch.no_grad()
    def test(self, test_dataset):
        print('dataset testing...')
        self.test_dataset = test_dataset
        test_pbar = tqdm(iter(self.test_dataset))
        i=0
        num_sample = 0
        print('computing quantative evaluations....')
        for gt_depth_uw, rgb_uw, ndepth_uw, conf_uw in test_pbar: 
            num_sample += gt_depth_uw.size(0)
            gt_depth_uw = gt_depth_uw.cuda()
            rgb_uw = rgb_uw.cuda()

            if 'RGBD' in self.mode:
                ndepth_uw = ndepth_uw.cuda()
            else:
                ndepth_uw = None

            conf_uw = conf_uw.cuda()

            model_input, test_metric_data ,tof_depth= \
                self._warp_test_data(gt_depth_uw, rgb_uw, ndepth_uw, conf_uw)

            depths_pred_uw = self.model(**model_input)
            test_metric_data['depth_pred_uw'] = depths_pred_uw[-1]

            monodepth_uw = self.midas(self.midas_transform(rgb_uw)).unsqueeze(1)
            monodepth_uw = monodepth_uw/monodepth_uw.view(monodepth_uw.size(0),-1).max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            monodepth_uw = 1 - monodepth_uw
            if self.resize4midas:
                monodepth_uw = F.interpolate(monodepth_uw, size=gt_depth_uw.shape[2:],
                                                mode="bicubic")
            # Only visualize and save the first batch
            if i == 0:
                for b in range(gt_depth_uw.size(0)):
                    img = rgb_uw[b]               # (C, H, W), device=cuda
                    vmin = img.amin()
                    vmax = img.amax()
                    print("min:", vmin.item(), "max:", vmax.item())
                    depth_img = tof_depth[b, 0]  # shape: [H, W]
                    vmin = depth_img.amin()
                    vmax = depth_img.amax()
                    print(f"ndepth_uw[{b}] min:", vmin.item(), "max:", vmax.item())
                    # 准备4张深度图: 输入warp后ndepth_uw，relative depth，预测深度，真实深度
                    imgs = {}
                    imgs['reprojected depth'] = self.depth_to_color(model_input['x_depth'][b,0])
                    imgs['relative depth'] = self.depth_to_color(monodepth_uw[b,0])
                    imgs['predicted depth'] = self.depth_to_color(depths_pred_uw[-1][b,0])
                    imgs['Ground Truth depth'] = self.depth_to_color(gt_depth_uw[b,0]/self.depth_scale)
                    # ====== Added: PSNR Input vs GT ======
                    input_depth = model_input['x_depth'][b,0]
                    gt_depth = gt_depth_uw[b,0]/self.depth_scale
                    mse_input = torch.mean((input_depth - gt_depth)**2)
                    max_val = gt_depth.max()
                    psnr_input = 10 * torch.log10((max_val**2) / mse_input)
                    print(f"[Sample {b}] Input vs GT PSNR: {psnr_input.item():.2f} dB")
                    # === Added: Compute and print PSNR (pred vs GT), only on the first batch ===
                    gt_meters   = (gt_depth_uw[b, 0] / self.depth_scale)           
                    pred_meters =  depths_pred_uw[-1][b, 0]                        

                    psnr_pred = self._compute_psnr(pred_meters, gt_meters, max_val=None)  # max_val=None → 自动取有效GT最大值
                    print(f"[PSNR] batch0 sample{b}: predicted vs GT = {psnr_pred:.2f} dB")

                    # for key in imgs:
                    #     imgs[key] = self.draw_lines(imgs[key], self.line_pos)

                    # Save the first four images and the depth profile plot (the fifth one)
                    depths_tensors = {
                        'reprojected depth': model_input['x_depth'][b,0],
                        'relative depth': monodepth_uw[b,0],
                        'predicted depth': depths_pred_uw[-1][b,0],
                        'Ground Truth depth': gt_depth_uw[b,0]/self.depth_scale
                    }

                    # Save the standalone depth profile plot
                    curve_path = os.path.join(self.save_dir, f'batch0_sample{b}_depth_lines.png')
                    self.plot_line_depths(depths_tensors, self.line_pos, curve_path)
                    self.plot_line_errors(depths_tensors, self.line_pos, self.save_dir, prefix=f"batch0_sample{b}_")

                    # Concatenate 5 images horizontally
                    gap = 20
                    w, h = imgs['reprojected depth'].size
                    total_w = w * 5 + gap * 4
                    combined_img = Image.new('RGB', (total_w, h), color=(255, 255, 255))
                    combined_img.paste(imgs['reprojected depth'], (0, 0))
                    combined_img.paste(imgs['relative depth'], (w + gap, 0))
                    combined_img.paste(imgs['predicted depth'], (2*(w + gap), 0))
                    combined_img.paste(imgs['Ground Truth depth'], (3*(w + gap), 0))
                    curve_img = Image.open(curve_path).resize((w, h))
                    combined_img.paste(curve_img, (4*(w + gap), 0))

                    imgs['reprojected depth'].save(f"{self.save_dir}/0_reprojected_depth{b}.png")
                    imgs['relative depth'].save(f"{self.save_dir}/1_relative_depth{b}.png")
                    imgs['predicted depth'].save(f"{self.save_dir}/2_predicted_depth{b}.png")
                    imgs['Ground Truth depth'].save(f"{self.save_dir}/3_gt_depth{b}.png")
                    draw = ImageDraw.Draw(combined_img)
                    font = None
                    try:
                        font = ImageFont.truetype("arial.ttf", 18)
                    except:
                        pass

                    save_path = os.path.join(self.save_dir, f'batch0_sample{b}_combined.png')
                    combined_img.save(save_path)
                    print(f'Saved visualization: {save_path}')

            if i==0:
                total_metrics = self._compute_test_metrics(**test_metric_data)
                i+=1
            elif self.test_dataset.batch_size != gt_depth_uw.size(0):
                i_ratio = gt_depth_uw.size(0)/self.test_dataset.batch_size
                metrics = self._compute_test_metrics(**test_metric_data)
                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]*i_ratio
                i+=i_ratio
            else:
                metrics = self._compute_test_metrics(**test_metric_data)
                for metric_name in metrics.keys():
                    total_metrics[metric_name] += metrics[metric_name]
                i+=1

        for metric_name in metrics.keys():
            metrics[metric_name] = total_metrics[metric_name]/i

        return metrics
    
