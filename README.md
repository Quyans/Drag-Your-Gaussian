<p align="center">
  <img width="15%" src="assets/logo1.png"/>
</p>

<p align="center">
<!--   <h1 align="center"><img height="100" src="https://github.com/imlixinyang/director3d-page/raw/master/assets/icon.ico"></h1> -->
  <h1 align="center"> Drag-Your-Gaussian</h1>
  <p align="center">
        <!-- <a href="[text](https://arxiv.org/pdf/2501.18672)"><img src='https://img.shields.io/badge/arXiv-DYG-red?logo=arxiv' alt='Paper PDF'></a> -->
        <a href="https://arxiv.org/abs/2501.18672"><img src='https://img.shields.io/badge/arXiv-DYG-red?logo=arxiv' alt='Paper PDF'></a>
        <a href='https://quyans.github.io/Drag-Your-Gaussian/'><img src='https://img.shields.io/badge/Project_Page-DYG-green' alt='Project Page'></a>
  </p>
  <p>Officially implement of the paper "Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting".</p>

## 😊 TL;DR

DYG allows users to drag 3D Gaussians, achieving flexible and precise 3D scene editing results.


## 🎥 Introduction Video

<!-- <p align="center">
  <img width="100%" src="assets/teaser.gif"/>
</p> -->

https://github.com/user-attachments/assets/1e484ff9-f44c-4995-a99d-453cf0f11f95



Visiting our [**Project Page**](https://quyans.github.io/Drag-Your-Gaussian/) for more result.

## 🔧 Installation
- clone this repo:
```
git clone https://github.com/Quyans/Drag-Your-Gaussian.git
cd Drag-Your-Gaussian
git submodule update --init --recursive 
```

- set up a new conda environment
```
conda env create --file environment.yaml
conda activate DYG
#install viser for webui
pip install -e ./viser
```

## 📚 Data Preparation
参考[3DGS](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes)进行重建，我们建议将重建过程的球谐阶数设为0 \
或者使用我们准备好的数据，以face场景为例，文件结构为
```
└── data
    └── face
        ├── image
        ├── sparse
        └── point_cloud.ply
```

由于我们使用lightningdrag作为扩散先验，您需要按照[lightningdrag](https://github.com/magic-research/LightningDrag/blob/main/INSTALLATION.md#2-download-pretrained-models)下载模型并按以下结构组织
```
└── checkpoints
    ├── dreamshaper-8-inpainting
    ├── lcm-lora-sdv1-5
    │   └── pytorch_lora_weights.safetensors
    ├── sd-vae-ft-ema
    │   ├── config.json
    │   ├── diffusion_pytorch_model.bin
    │   └── diffusion_pytorch_model.safetensors
    ├── IP-Adapter/models
    │   ├── image_encoder
    │   └── ip-adapter_sd15.bin
    └── lightning-drag-sd15
        ├── appearance_encoder
        │   ├── config.json
        │   └── diffusion_pytorch_model.safetensors
        ├── point_embedding
        │   └── point_embedding.pt
        └── lightning-drag-sd15-attn.bin
```

## 🚋 Training
使用以下命令启动webui，具体可参考[webui](./assets/webui-guide/webui.md)
```
python webui.py --colmap_dir <path to colmap dir> --gs_source <path to 3DGS ply> --output_dir <save path>
#这是一个具体的例子
python webui.py --colmap_dir ./data/face/ --gs_source ./data/face/point_cloud.ply --output_dir result
```
您可以在webui中直接进行训练。或者在webui选择drag point和mask后，导出对应数据drag_point.json、gaussian_mask.pt，通过以下命令开始训练，其中--point_dir和--mask_dir参数代表对应的文件路径
```
python drag_3d.py --config configs/main.yaml \
                  --colmap_dir ./data/face/ \
                  --gs_source ./data/face/point_cloud.ply \
                  --point_dir ./data/face/export/drag_points.json \
                  --mask_dir ./data/face/export/gaussian_mask.pt \
                  --output_dir result
```