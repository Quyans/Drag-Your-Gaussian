
<p align="center">
  <img width="15%" src="assets/logo1.png"/>
</p>

<h1 align="center">[SIGGRAPH 2025] Drag-Your-Gaussian</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2501.18672"><img src="https://img.shields.io/badge/arXiv-DYG-red?logo=arxiv" alt="Paper PDF"></a>
  <a href="https://quyans.github.io/Drag-Your-Gaussian/"><img src="https://img.shields.io/badge/Project_Page-DYG-green" alt="Project Page"></a>
</p>

<p align="center">
  Official implementation of the paper:<br>
  <strong>“Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting.”</strong>
</p>

---

## 😊 TL;DR

**DYG** allows intuitive and flexible 3D scene editing by enabling users to drag 3D Gaussians while preserving fidelity and structure.

---

## 🎥 Introduction Video

Watch our intro video:  
https://github.com/user-attachments/assets/1e484ff9-f44c-4995-a99d-453cf0f11f95

Or visit our [**Project Page**](https://quyans.github.io/Drag-Your-Gaussian/) for more examples and visualizations.

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/Quyans/Drag-Your-Gaussian.git
cd Drag-Your-Gaussian
git submodule update --init --recursive 
```

Create a new conda environment:

```bash
conda env create --file environment.yaml
conda activate DYG
```

---

## 📚 Data Preparation

Follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) for reconstruction.  
We recommend setting the **spherical harmonic degree to 0**.

Alternatively, you can use our prepared [example data](https://drive.google.com/drive/folders/19Jv3crbF7xMu1ouNoCH-mEH87ClykpuY?usp=sharing).  
Example structure (e.g., `face` scene):

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

### 🔄 Diffusion Prior

We use LightningDrag as the diffusion prior. Follow [LightningDrag Installation Guide](https://github.com/magic-research/LightningDrag/blob/main/INSTALLATION.md#2-download-pretrained-models) to download required models.  
Organize them as follows:

```
└── checkpoints
    ├── dreamshaper-8-inpainting
    ├── lcm-lora-sdv1-5/
    │   └── pytorch_lora_weights.safetensors
    ├── sd-vae-ft-ema/
    │   ├── config.json
    │   ├── diffusion_pytorch_model.bin
    │   └── diffusion_pytorch_model.safetensors
    ├── IP-Adapter/models/
    │   ├── image_encoder
    │   └── ip-adapter_sd15.bin
    └── lightning-drag-sd15/
        ├── appearance_encoder/
        │   ├── config.json
        │   └── diffusion_pytorch_model.safetensors
        ├── point_embedding/
        │   └── point_embedding.pt
        └── lightning-drag-sd15-attn.bin
```

---

## 🚋 Training

### 🖥️ WebUI

Launch the WebUI:

```bash
python webui.py --colmap_dir <path_to_colmap> --gs_source <path_to_pointcloud.ply> --output_dir <save_path>
```

**Example:**

```bash
python webui.py --colmap_dir ./data/face/ --gs_source ./data/face/point_cloud.ply --output_dir result
```

You can train directly in the WebUI. Alternatively, after selecting drag points and masks, export the files and run:

```bash
python drag_3d.py --config configs/main.yaml                   --colmap_dir ./data/face/                   --gs_source ./data/face/point_cloud.ply                   --point_dir ./data/face/export_1/drag_points.json                   --mask_dir ./data/face/export_1/gaussian_mask.pt                   --output_dir result
```

---

## 📖 Citation

If you find our work useful, please cite:

```bibtex
@article{qu2025drag,
  title={Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting},
  author={Qu, Yansong and Chen, Dian and Li, Xinyang and Li, Xiaofan and Zhang, Shengchuan and Cao, Liujuan and Ji, Rongrong},
  journal={arXiv preprint arXiv:2501.18672},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the **CC BY-NC-SA 4.0**.  
The code is intended for **academic research purposes only**.

---

## 📬 Contact

For any questions or collaborations, feel free to contact:  
📧 [quyans@stu.xmu.edu.cn](mailto:quyans@stu.xmu.edu.cn)
