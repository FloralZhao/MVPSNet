from pytorch3d.loss import chamfer_distance
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.ops.points_alignment import iterative_closest_point, _apply_similarity_transform
from pytorch3d.io import save_ply, save_obj

import torch

import trimesh
import numpy as np
import json
import os
import argparse

from IPython.core.debugger import set_trace

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:2")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstruction_path', type=str, required=True)
    parser.add_argument('--ground_truth', type=str, help='path to ground truth meshes.')

    args = parser.parse_args()
    return args

def load_verts(filename):
    mesh = trimesh.load(filename)
    verts = mesh.vertices
    verts = verts[verts[:, -1] >= 5]
    verts = torch.from_numpy(verts).float().unsqueeze(0).to(device)
    return verts

if __name__ == "__main__":
    args = parse_args()
    reconstruction_path = args.reconstruction_path
    gt_path = args.ground_truth
    cds = []
    cds_icp = []
    for obj in ['bear', 'buddha', 'cow', 'pot2', 'reading']:
        print(f'\t {obj}')
        # load meshes
        verts_gt, _ = load_ply(os.path.join(gt_path, f'{obj}PNG', f'Groundtruth/{obj}PNG_Mesh_GT.ply'))
        verts_gt = verts_gt[verts_gt[:, -1] >= 5]
        verts_gt = verts_gt.unsqueeze(0).to(device)

        verts_ours = load_verts(
            os.path.join(reconstruction_path, f'{obj}.ply'))
        verts_ours_icp = iterative_closest_point(verts_ours, verts_gt).Xt

        # Compute Chamfer distance.
        cd, n = chamfer_distance(verts_ours, verts_gt, norm=1)
        print(f'\t \t Ours: {cd}')
        cds.append(cd.item())

        cd, n = chamfer_distance(verts_ours_icp, verts_gt, norm=1)
        print(f'\t \t Ours_ICP: {cd}')
        cds_icp.append(cd.item())

    print(f'\t average: {np.mean(cds)}')
    print(f'\t average after ICP: {np.mean(cds_icp)}')

