# MVPSNet: Fast Generalizable Multi-view Photometric Stereo

### [Project Page](https://floralzhao.github.io/mvpsnet.github.io/) | [Paper](https://arxiv.org/abs/2305.11167)

## Getting the data
### sMVPS

The dataset is arond 200 GB. Please fill out this [form](https://forms.gle/Yq1YiSGAFnrTYsYE8) to request access to our sMVPS dataset on the shared cloud storage. You'll have to be manually added to the viewer list.

Thanks to [Daniel Lichy](https://www.cs.umd.edu/~dlichy/) for generating the data.

### DiLiGenT-MV
You can download the DiLiGenT-MV dataset from their [website](https://sites.google.com/site/photometricstereodata/mv).

## Usage
### Setup
We highly recommend using conda to setup a virual environment. Then run `pip -r requirements.txt` to add dependencies.

You can download the pre-trained model [here](https://drive.google.com/file/d/1FtAAztfJnHJkHAqElMISYaugMrMFFwfv/view?usp=sharing).
### Generate meshes
Run the following command to generate 3D point clouds.

```
DILIGENT_MV=...  # update the path of DiLiGenT_MV dataset
CKPT_FILE=...  # update the path of checkpoint
MESH_FOLDER=...  # path to save the meshes
python run.py --dataset=diligent_mv --testpath=$DILIGENT_MV --loadckpt=$CKPT_FILE --save_folder=$MESH_FOLDER --numlights=10 --numviews=5 $@ 
```

Then use [MeshLab](https://www.meshlab.net/) to reconstruct meshes from point clouds using Screened Poisson method.


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
