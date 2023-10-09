import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models import *
from models.cas_mvpsnet import CascadeMVSNet
from models.SDPS_Net4 import NENet
from utils.utils import *
from utils.console_logger import ConsoleLogger
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import scipy.io as sio
import math



cudnn.benchmark = True
cudnn.deterministc = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse.')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')

# PS model
parser.add_argument('--numlights', type=int, default=20)
parser.add_argument('--numviews', type=int, default=3)
parser.add_argument('--ps_fuse_type', default='max', type=str)
parser.add_argument('--ps_feat_chs', type=int, default=16)
# MVS model
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default=None)
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--use_sdps', action='store_true', help='use lighting directions predicted by SDPS-Net.')

# filtering
parser.add_argument('--geo_mask_thresh', default=2, type=int, help="used in geo_mask")
parser.add_argument('--photo_mask_thresh', default=0.1, type=float, help="used in photo_mask")

parser.add_argument('--save_folder', default=None, type=str)

# parse arguments and check
args = parser.parse_args()

if args.loadckpt is not None:
    logdir = os.path.dirname(os.path.dirname(args.loadckpt))
    LOGGER = ConsoleLogger(phase='eval', logfile_dir=logdir)
    logdir = LOGGER.getLogFolder()
else:
    logdir = args.logdir
    LOGGER = ConsoleLogger(phase='eval', abs_logdir=logdir)

LOGGER.info(f"argv: {sys.argv[1:]}")
log_args(args, LOGGER)


# read an image
def read_img(filename):
    # img = Image.open(filename)
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # scale 0~65535 to 0~1
    # img = img.astype(np.float32) / 65535.
    img = (img.astype(np.float32) - np.min(img)) / (np.max(img) - np.min(img))  # 0~1
    img = img[:, 50:562, :]
    return img


# read a binary mask
def read_mask(filename):
    mask = np.array(Image.open(filename), dtype=np.float32)
    if np.max(mask) == 255:
        mask = mask / 255.
    mask = mask[:, 50:562]
    return mask > 0.5

# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def read_normal(filename, downscale):
    normal = sio.loadmat(filename)['Normal_gt']
    normal = normal[:, 50:562, :]
    norm = np.sqrt((normal * normal).sum(2, keepdims=True))
    normal = normal / (norm + 1e-10)
    # normal = 0.5 * normal + 0.5
    normal = normal[::downscale, ::downscale, :]
    return normal


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1:]]
            data.append((ref_view, src_views))
    return data


def process_cam_params(intrinsics, R, t):
    intrinsics[0][2] -= 50
    # intrinsics[:2, :] = intrinsics[:2, :] / 4
    extrinsics = np.concatenate((R, t), axis=-1)
    extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), 0).astype(np.float32)
    return intrinsics, extrinsics


def mae(a, b, mask):
    angular_error = np.arccos(np.sum(a * b, 2).clip(-1, 1))
    angular_error_deg = (180.0 / np.pi) * angular_error
    mask = mask > 0.5
    mae = np.mean(angular_error_deg[mask])
    return mae

def mae_ps_nerf(vec1, vec2, mask=None, normalize=True):
    '''
    Input : N x 3  or  H x W x 3 .   [-1,1]
    Output : MAE, AE
    '''
    vec1, vec2 = vec1.copy(), vec2.copy()
    mask = mask.copy() if mask is not None else mask
    if normalize:
        norm1 = np.linalg.norm(vec1.astype(np.float64), axis=-1)
        norm2 = np.linalg.norm(vec2.astype(np.float64), axis=-1)
        vec1 /= norm1[..., None] + 1e-5
        vec2 /= norm2[..., None] + 1e-5
        vec1[norm1 == 0] = 0
        vec2[norm2 == 0] = 0
    dot_product = (vec1.astype(np.float64) * vec2.astype(np.float64)).sum(-1).clip(-1, 1)
    if mask is not None:
        dot_product = dot_product[mask.astype(bool)]
    angular_err = np.arccos(dot_product) * 180.0 / math.pi
    l_err_mean = angular_err.mean()
    return l_err_mean


# run MVS model to save depth maps and confidence maps
def save_depth():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, "test", args.numviews, args.numdepth, args.numlights, use_sdps=args.use_sdps)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.batch_size, drop_last=False,
                               pin_memory=True)

    # model
    ps_model = NENet(base_chs=args.ps_feat_chs, fuse_type=args.ps_fuse_type, c_in=6)
    mvs_model = CascadeMVSNet(cr_in_chs=[args.ps_feat_chs * 4, args.ps_feat_chs * 2, args.ps_feat_chs], refine=False,
                              ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                              depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                              share_cr=args.share_cr,
                              cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                              grad_method=args.grad_method)
    ps_model = nn.DataParallel(ps_model)
    mvs_model = nn.DataParallel(mvs_model)
    ps_model.cuda()
    mvs_model.cuda()

    # load checkpoint file specified by args.loadckpt
    LOGGER.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    ps_model.load_state_dict(state_dict['ps_model'])
    mvs_model.load_state_dict(state_dict['mvs_model'])
    ps_model.eval()
    mvs_model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            outputs = {}
            feats, outputs['normal'] = ps_model(sample_cuda)
            mvs_outputs = mvs_model(sample_cuda['imgs'], feats,
                                    sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs.update(mvs_outputs)
            outputs = tensor2numpy(outputs)
            del sample_cuda
            LOGGER.info('Iter {}/{}'.format(batch_idx, len(TestImgLoader)))
            filenames = sample["filename"]

            # save depth maps and confidence maps and normal
            for filename, depth_est, normal_est, photometric_confidence in zip(filenames, outputs['stage3']["depth"],
                                                                    outputs['normal'][:,0],
                                                                   outputs['stage3']["photometric_confidence"]):
                depth_filename = os.path.join(logdir, filename.format('depth_est', '.pfm'))
                normal_filename = os.path.join(logdir, filename.format('normal_est', '.pfm'))
                confidence_filename = os.path.join(logdir, filename.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(normal_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)

                # save depth maps
                save_pfm(depth_filename, depth_est)
                # save normal map
                normal_est = normal_est.transpose(1,2,0)
                save_pfm(normal_filename, normal_est)
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scan_folder, out_folder, scene, save_folder=None):
    # the pair file
    pair_file = os.path.join(args.testpath, "diligent_mv_pairs.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []
    vertex_normals = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # read camera parameters
    calib = sio.loadmat(os.path.join(scan_folder, 'Calib_Results.mat'))
    for ref_view, src_views in pair_data:
        # load the camera parameters
        intrinsics = calib['KK'].astype(np.float32).copy()
        ref_R = calib['Rc_%d' % (ref_view+1)].astype(np.float32).copy()
        t = calib['Tc_%d' % (ref_view+1)].astype(np.float32).copy()
        ref_intrinsics, ref_extrinsics = process_cam_params(intrinsics, ref_R, t)

        # load the reference image. Use light 3
        ref_img = read_img(os.path.join(scan_folder, f'view_{ref_view+1:02d}', '003.png'))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>2}.pfm'.format(ref_view)))[0]
        # load the estimated normal of the reference view
        ref_normal_est = read_pfm(os.path.join(out_folder, 'normal_est/{:0>2}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>2}.pfm'.format(ref_view)))[0]
        photo_mask = confidence > args.photo_mask_thresh

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            intrinsics = calib['KK'].astype(np.float32).copy()
            R = calib['Rc_%d' % (src_view + 1)].astype(np.float32).copy()
            t = calib['Tc_%d' % (src_view + 1)].astype(np.float32).copy()
            src_intrinsics, src_extrinsics = process_cam_params(intrinsics, R, t)
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>2}.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= args.geo_mask_thresh

        # object mask
        object_mask = read_mask(os.path.join(scan_folder, 'mask_depth', f'view_{ref_view+1:02d}.png'))
        # final_mask = np.logical_and(photo_mask, geo_mask)
        final_mask = np.logical_and(object_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>2}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>2}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>2}_final.png".format(ref_view)), final_mask)

        LOGGER.info("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        if args.display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(0)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        LOGGER.info(f"valid_points: {valid_points.mean()}")
        x, y, depth, normal = x[valid_points], y[valid_points], depth_est_averaged[valid_points], ref_normal_est[valid_points]
        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]  # (N, 3)
        vertexs.append(xyz_world.transpose((1, 0)))  # (3, N)
        vertex_colors.append((color * 255).astype(np.uint8))
        normal_world = np.matmul(normal, ref_R)
        vertex_normals.append(normal_world)

        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertex_normals = np.concatenate(vertex_normals, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_normals = np.array([tuple(v) for v in vertex_normals], dtype=[('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

    vertex_all_with_normal = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr + vertex_normals.dtype.descr)
    for prop in vertexs.dtype.names:  # ['x', 'y', 'z']
        vertex_all_with_normal[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all_with_normal[prop] = vertex_colors[prop]
    for prop in vertex_normals.dtype.names:
        vertex_all_with_normal[prop] = vertex_normals[prop]

    el = PlyElement.describe(vertex_all_with_normal, 'vertex')
    plyfilename =  os.path.join(out_folder, 'casmvps_{}.ply'.format(scene))
    PlyData([el]).write(plyfilename)
    LOGGER.info(f"saving the final model to {plyfilename}")

    if save_folder is not None:
        PlyData([el]).write(os.path.join(save_folder, 'casmvps_{}.ply'.format(scene)))


    vertex_all_no_normal = np.empty(len(vertexs),
                          vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:  # ['x', 'y', 'z']
        vertex_all_no_normal[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all_no_normal[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all_no_normal, 'vertex')
    plyfilename = os.path.join(out_folder, 'casmvps_{}_no_normal.ply'.format(scene))
    PlyData([el]).write(plyfilename)
    LOGGER.info(f"saving the final model to {plyfilename}")

    if save_folder is not None:
        PlyData([el]).write(os.path.join(save_folder, 'casmvps_{}_no_normal.ply'.format(scene)))



def step1():
    save_depth()


def step2(save_folder):
    for scene in ['bearPNG', 'buddhaPNG', 'cowPNG', 'pot2PNG', 'readingPNG']:
        scan_folder = os.path.join(args.testpath, scene)
        out_folder = os.path.join(logdir, scene)
        # step2. filter saved depth maps with photometric confidence maps and geometric constraints
        filter_depth(scan_folder, out_folder, scene, save_folder)



if __name__ == '__main__':
    if args.save_folder is not None:
        os.makedirs(args.save_folder, exist_ok=True)

    starting_time = time.time()
    # step1. save all the depth maps and the masks in outputs directory
    step1()

    # step 2: 3d metrics
    step2(args.save_folder)

    endding_time = time.time()

    LOGGER.info(f'Test time for 5 objects: {endding_time-starting_time}')

