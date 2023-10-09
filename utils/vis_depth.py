import numpy as np
import os
import matplotlib.pyplot as plt
from .utils import *

cmap = plt.cm.jet
def colored_depth(depth, vmin, vmax):
    """
        depth: (H, W)
        """
    if type(depth) is not np.ndarray:
        depth = depth.detach().cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if vmin is None:
        vmin = np.min(x[x > 0])  # get minimum positive depth (ignore background)
    if vmax is None:
        vmax = np.max(x)

    x = (x - vmin) / (vmax - vmin + 1e-8)  # normalize to 0~1
    return cmap(x)[:,:,:3]


########## visualization for train_cas.py
def detailed_depth_image_summary_cas(outputs, depth_gt_ms, mask_ms, image_outputs, near, far):
    '''outputs: dict'''
    for key in ['stage1', 'stage2', 'stage3']:
        # vis depth
        depth_est_vis = outputs[key]['depth']
        depth_est_vis = colored_depth(depth_est_vis[0].detach().cpu().numpy().copy(), near, far)
        depth_est_vis = depth_est_vis * mask_ms[key][0].detach().cpu().numpy().copy()[..., None]  # (h, w, 3) np array
        image_outputs[f'depth_est_{key}'] = torch.from_numpy(depth_est_vis).permute(2, 0, 1).unsqueeze(0)  # add batch dim
        depth_gt_vis = depth_gt_ms[key]
        depth_gt_vis = colored_depth(depth_gt_vis[0].detach().cpu().numpy().copy(), near, far)
        depth_gt_vis = depth_gt_vis * mask_ms[key][0].detach().cpu().numpy().copy()[..., None]
        image_outputs[f'depth_gt_{key}'] = torch.from_numpy(depth_gt_vis).permute(2, 0, 1).unsqueeze(0)

        errormap = (outputs[key]['depth'] - depth_gt_ms[key]).abs()
        error_map_vis = colored_depth(errormap[0].detach().cpu().numpy().copy(), 0, far - near)
        error_map_vis = error_map_vis * mask_ms[key][0].detach().cpu().numpy().copy()[..., None]  # (h, w, 3) np array
        image_outputs[f'error_map_{key}'] = torch.from_numpy(error_map_vis).permute(2, 0, 1).unsqueeze(0)


def detailed_depth_scalar_summary_cas(outputs, depth_gt_ms, mask_ms, scalar_outputs):
    '''outputs: dict'''
    for key in ['stage1', 'stage2', 'stage3']:
        scalar_outputs[f'abs_depth_error_{key}'] = AbsDepthError_metrics(outputs[key]['depth'], depth_gt_ms[key],
                                                                         mask_ms[key] > 0.5)
        scalar_outputs[f'thres2mm_error_{key}'] = Thres_metrics(outputs[key]['depth'], depth_gt_ms[key],
                                                                mask_ms[key] > 0.5, 2)
        scalar_outputs[f'thres4mm_error_{key}'] = Thres_metrics(outputs[key]['depth'], depth_gt_ms[key], mask_ms[key] > 0.5, 4)
        scalar_outputs[f'thres8mm_error_{key}'] = Thres_metrics(outputs[key]['depth'], depth_gt_ms[key], mask_ms[key] > 0.5, 8)


def detailed_depth_image_save(image_outputs, vis_dir, batch_idx, near, far):
    for key in ['stage1', 'stage2', 'stage3']:
        depth_est = image_outputs[f'depth_est_{key}']
        depth_gt = image_outputs[f'depth_gt_{key}']
        errormap = image_outputs[f'errormap_{key}']
        depth_est_filename = os.path.join(vis_dir, f'depth_est_batch_{batch_idx}_{key}.png')
        depth_gt_filename = os.path.join(vis_dir, f'depth_gt_batch_{batch_idx}_{key}.png')
        errormap_filename = os.path.join(vis_dir, f'depth_errormap_batch_{batch_idx}_{key}.png')
        plt.imshow(depth_est[0].cpu().numpy(), vmin=near, vmax=far, cmap='jet')
        plt.colorbar()
        plt.savefig(depth_est_filename)
        plt.close()
        plt.imshow(depth_gt[0].cpu().numpy(), vmin=near, vmax=far, cmap='jet')
        plt.colorbar()
        plt.savefig(depth_gt_filename)
        plt.close()
        plt.imshow(errormap[0].cpu().numpy(), vmin=0, vmax=8, cmap='jet')
        plt.colorbar()
        plt.savefig(errormap_filename)
        plt.close()


########## visualization for train_mvps_cas.py
def detailed_depth_image_summary_mvps_cas(outputs, depth_gt_ms, mask_ms, image_outputs, near, far):
    '''outputs: dict'''
    for key in ['stage1', 'stage2', 'stage3']:
        # vis depth
        depth_est_vis = outputs[key]['depth']
        depth_est_vis = colored_depth(depth_est_vis[0].detach().cpu().numpy().copy(), near, far)
        depth_est_vis = depth_est_vis * mask_ms[key][0][0].detach().cpu().numpy().copy()[..., None]  # (h, w, 3) np array
        image_outputs[f'depth_est_{key}'] = torch.from_numpy(depth_est_vis).permute(2, 0, 1).unsqueeze(0)  # add batch dim
        depth_gt_vis = depth_gt_ms[key]
        depth_gt_vis = colored_depth(depth_gt_vis[0].detach().cpu().numpy().copy(), near, far)
        depth_gt_vis = depth_gt_vis * mask_ms[key][0][0].detach().cpu().numpy().copy()[..., None]
        image_outputs[f'depth_gt_{key}'] = torch.from_numpy(depth_gt_vis).permute(2, 0, 1).unsqueeze(0)

        errormap = (outputs[key]['depth'] - depth_gt_ms[key]).abs()
        error_map_vis = colored_depth(errormap[0].detach().cpu().numpy().copy(), 0, 2)
        error_map_vis = error_map_vis * mask_ms[key][0][0].detach().cpu().numpy().copy()[..., None]  # (h, w, 3) np array
        image_outputs[f'error_map_{key}'] = torch.from_numpy(error_map_vis).permute(2, 0, 1).unsqueeze(0)


def detailed_depth_scalar_summary_mvps_cas(outputs, depth_gt_ms, mask_ms, scalar_outputs):
    '''outputs: dict'''
    for key in ['stage1', 'stage2', 'stage3']:
        scalar_outputs[f'abs_depth_error_{key}'] = AbsDepthError_metrics(outputs[key]['depth'], depth_gt_ms[key],
                                                                         mask_ms[key][:, 0] > 0.5)
        scalar_outputs[f'thres2mm_error_{key}'] = Thres_metrics(outputs[key]['depth'], depth_gt_ms[key],
                                                                mask_ms[key][:, 0] > 0.5, 2)
        scalar_outputs[f'thres4mm_error_{key}'] = Thres_metrics(outputs[key]['depth'], depth_gt_ms[key], mask_ms[key][:, 0] > 0.5, 4)
        scalar_outputs[f'thres8mm_error_{key}'] = Thres_metrics(outputs[key]['depth'], depth_gt_ms[key], mask_ms[key][:, 0] > 0.5, 8)
