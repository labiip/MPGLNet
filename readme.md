# MPGLNet
* This project is the official implementation for MPGLNet.

# Requirements
Some important required packages include:
* Pytorch >=1.7.0.
* Python >= 3.7
* CUDA >= 9.0
* GCC >= 4.9
* torchvision
* timm
* open3d
* TensorBoardX
* Some basic python packages such as argparse, easydict, opencv-python, tqdm......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Building Pytorch Extensions for Chamfer Distance, PointNet++ and KNN
```
# Chamfer Distance
bash install.sh
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
2. Inference
```
python tools/inference.py \
${POINTR_CONFIG_FILE} ${POINTR_CHECKPOINT_FILE} \
[--pc_root <path> or --pc <file>] \
[--save_vis_img] \
[--out_pc_root <dir>] \
```
3. Evaluation
```
bash ./scripts/test.sh <GPU_IDS>  \
    --ckpts <path> \
    --config <config> \
    --exp_name <name>
```

4. Training
```
bash ./scripts/train.sh <GPUIDS> \
    --config <config> \
    --exp_name <name>
```

# Acknowledgement
* The codebase is adapted from the work [PoinTr](https://github.com/yuxumin/PoinTr), [CRAPCN](https://github.com/EasyRy/CRA-PCN). We thank for their excellent works.