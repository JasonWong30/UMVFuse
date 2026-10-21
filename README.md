
# UMVFuse

## Toward Unified Multi-modal Image and Video Fusion via Dual-Teacher Prior Distillation

Our paper, **"Toward Unified Multi-modal Image and Video Fusion via Dual-Teacher Prior Distillation,"** is now available as an Early Access article in *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)*.

Some parts of the code are still being organized and refined. The complete implementation, pretrained models, documentation, and related materials will therefore be released progressively. This repository will continue to be updated as additional resources become available.

## 📎 Project & Resources

| Resource | Status | Link |
|----------|--------|------|
| 📄 Paper | Early Access (IEEE TCSVT) | IEEE Xplore link to be added |
| 📊 Datasets | Released | See Dataset Release Status ↓ |
| 🔑 Code | Staged Rollout | See Code and Model Release Status ↓ |
| 💻 Performance | Staged Rollout | See Qualitative and Quantitative Release Status ↓ |

## 📢 Update Log

Major updates to the repository will be recorded here in reverse chronological order.

| Date | Update | Description |
|------|--------|-------------|
| 2026-10-21 | 📄 Paper & README | Updated the publication status to IEEE TCSVT Early Access, revised the citation, and added the update log. |

## 📊 Dataset Release Status

The BraTS2020 sequence dataset is available at [this link](https://www.med.upenn.edu/cbica/brats2020/data.html).

The M3FD test sequences are available at [this link](https://pan.baidu.com/s/1ih94fQQql0BxU1RhaZphTQ?pwd=QWER). (Note: We manually constructed eight short sequences from temporally continuous image pairs.)

The Polarization dataset is available at [this link](http://www.ok.sc.e.titech.ac.jp/res/PolarDem/index.html).

## 🔑 Code and Model Release Status

### Training

#### Fusion Stage

```bash
torchrun --nproc_per_node=2 main_fusepro.py \
    --checkpoint /path/to/pretrained_encoder.pth \
    --dataset_config_path options/fusion.yml
```

### Inference

```bash
python test_fuse.py \
    --checkpoint /path/to/fusion_checkpoint.pth \
    --dataset_config_path options/fusion.yml \
    --save_images \
    --fuse_vi_y_channel
```

## 💻 Performance Release Status

### Qualitative comparison of different methods in scenarios with abrupt changes in local content

<p align="center">
  <img src="asset/局部场景突变.jpg" width="70%" alt="Qualitative comparison under abrupt local content changes" />
</p>

### Visualization of inter-frame difference maps in scenarios with large-scale and non-uniform motion

<p align="center">
  <img src="asset/Temporal_Vis.jpg" width="70%" alt="Inter-frame difference map comparison under large-scale and non-uniform motion" />
</p>

## 📑 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@article{Huang2026UMVFuse,
  title   = {Toward Unified Multi-modal Image and Video Fusion via Dual-Teacher Prior Distillation},
  author  = {Jingxue Huang and Xincheng Yu and Xiaosong Li and Yi Lin},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026},
  note    = {Early Access}
}
```

*The DOI and final volume, issue, and page numbers will be added when available.*
