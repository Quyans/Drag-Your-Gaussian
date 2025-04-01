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
```

## 📚 Data Preparation

Refer to [3DGS](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) for reconstruction. We recommend setting the spherical harmonic degree to 0 for the reconstruction process.
Alternatively, you can use the [data](https://drive.google.com/drive/folders/19Jv3crbF7xMu1ouNoCH-mEH87ClykpuY?usp=sharing) we have prepared. Taking the face scene as an example, the file structure is as follows:
```
└── data
    └── face
        ├── export_1
        │   ├── drag_points.json
        │   └── gaussian_mask.pt
        ├── image
        ├── sparse
        └── point_cloud.ply
```
Since we use LightningDrag as the diffusion prior, you need to download the model following the instructions in [lightningdrag](https://github.com/magic-research/LightningDrag/blob/main/INSTALLATION.md#2-download-pretrained-models) and organize it according to the following structure.
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
Start the WebUI using the following command. For detailed usage instructions, refer to [WebUI](./assets/webui-guide/webui.md)
```shell
python webui.py --colmap_dir <path to colmap dir> --gs_source <path to 3DGS ply> --output_dir <save path>
#This is a specific example.
python webui.py --colmap_dir ./data/face/ --gs_source ./data/face/point_cloud.ply --output_dir result
```
You can train directly in the WebUI. Alternatively, after selecting the drag point and mask in the WebUI, export the corresponding data files drag_point.json and gaussian_mask.pt, and start the training using the following command, where the `--point_dir` and `--mask_dir` parameters represent the file paths.

```shell
python drag_3d.py --config configs/main.yaml \
                  --colmap_dir ./data/face/ \
                  --gs_source ./data/face/point_cloud.ply \
                  --point_dir ./data/face/export_1/drag_points.json \
                  --mask_dir ./data/face/export_1/gaussian_mask.pt \
                  --output_dir result
```

## Citation

```
@article{qu2025drag,
  title={Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting},
  author={Qu, Yansong and Chen, Dian and Li, Xinyang and Li, Xiaofan and Zhang, Shengchuan and Cao, Liujuan and Ji, Rongrong},
  journal={arXiv preprint arXiv:2501.18672},
  year={2025}
}
```

## License

Licensed under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)


The code is released for academic research use only. 

If you have any questions, please contact me via [quyans@stu.xmu.edu.cn](mailto:quyans@stu.xmu.edu.cn). 