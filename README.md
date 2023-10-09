# MVPSNet: Fast Generalizable Multi-view Photometric Stereo

### [Project Page](https://floralzhao.github.io/mvpsnet.github.io/) | [Paper](https://arxiv.org/abs/2305.11167)

## Getting the data
### sMVPS

The dataset is ~ 200 GB on OneDrive. Please fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSczkW6A4UsmbjHPssQ_AS3zaYRgXnRti1PhoGWkeR_mJeH4Lw/viewform?usp=sf_link) to request access. You'll have to be manually added to the viewer list.

Thanks to [Daniel Lichy](https://www.cs.umd.edu/~dlichy/) for generating the data.

### DiLiGenT-MV
You can download the DiLiGenT-MV dataset from their [website](https://sites.google.com/site/photometricstereodata/mv).

## Usage
### Setup
We highly recommend using conda to setup a virual environment. Then run `pip install -r requirements.txt` to add dependencies.

You can download the pre-trained model [here](https://drive.google.com/file/d/1FtAAztfJnHJkHAqElMISYaugMrMFFwfv/view?usp=sharing).
### Generate meshes
Run the following command to generate 3D point clouds.

```
export DILIGENT_MV=...  # update the path of DiLiGenT_MV dataset
export CKPT_FILE=...  # update the path of checkpoint
export MESH_FOLDER=...  # path to save the meshes
python run.py --dataset=diligent_mv --testpath=$DILIGENT_MV --loadckpt=$CKPT_FILE --save_folder=$MESH_FOLDER --numlights=10 --numviews=5 $@ 
```

Then use [MeshLab](https://www.meshlab.net/) to reconstruct meshes from point clouds using Screened Poisson method.

### Evaluation
Our evaluation metrics were implemented based on [pytorch3d](https://pytorch3d.org/). You can install it following their [instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). 

You also need to install [trimesh](https://github.com/mikedh/trimesh) by running
```
pip install trimesh
```

Then run the following commands:
```
export MESH_PATH={your mesh path}
export GT_PATH={ground truth meshes path}
python evaluate.py --reconstruction_path=$MESH_PATH --ground_truth=$GT_PATH
```

## Citation

```
@inproceedings{zhao2023mvpsnet,
  title={MVPSNet: Fast Generalizable Multi-view Photometric Stereo},
  author={Zhao, Dongxu and Lichy, Daniel and Perrin, Pierre-Nicolas and Frahm, Jan-Michael and Sengupta, Soumyadip},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12525--12536},
  year={2023}
}
```

# Acknowledgements

Parts of the code were based on CasMVSNet: https://github.com/hz-ants/cascade-mvsnet/tree/master/CasMVSNet, SDPS-Net https://github.com/guanyingc/SDPS-Net and ShapeAndMaterial https://github.com/dlichy/ShapeAndMaterial.
