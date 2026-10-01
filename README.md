# UltraCCM

This repository contains the official implementation of UltraCCM introduced in
the following paper:

[Medical Image Analysis] [Micro-ultrasound super-resolution with geometry-driven consistency models](https://doi.org/10.1016/j.media.2026.104333)
<br>
[Kaifeng Pang](https://kfpang.com), [Kai Zhao](https://kaizhao.net/), [Qi Miao](https://mrrl.ucla.edu/people/qi-miao-md-phd), [Alex Ling Yu Hung](https://web.cs.ucla.edu/~alexhung/), [Changsuk Oh](https://mrrl.ucla.edu/people/changsuk-oh-phd), [Raymi Ramirez](https://www.uclahealth.org/departments/radonc/education/residents/raymi-ramirez-phd), [Qiudi He](https://mrrl.ucla.edu/people/qiudi-he), Jordan Klein, [Wei Shao](https://www.linkedin.com/in/wei-shao-438782115/), [Wayne Brisbane](https://www.uclahealth.org/providers/wayne-brisbane), [Kyunghyun Sung](https://mrrl.ucla.edu/people/kyung-sung-phd)
<br>
Medical Image Analysis, 2026

## Installation

The code was developed with Python 3.10 and CUDA-enabled PyTorch. Install an
MPI implementation first, then install the PyTorch build appropriate for your
CUDA version. Finally install this repository in editable mode:

```bash
git clone <repository-url>
cd UltraCCM
python -m venv .venv
source .venv/bin/activate

# Install PyTorch/torchvision for your CUDA version first.
pip install -e .
```

All commands below use one MPI process. Multi-GPU training can be launched by
increasing `-n` and choosing a global batch size divisible by the process count.

## Data format

Each case is a directory containing one DICOM series:

```text
data/
  invivo/
    train/case_001/*.dcm
    validation/case_003/*.dcm
    test/case_004/*.dcm
```

The loader expects each DICOM to expose `pixel_array`, `PixelSpacing`, and
`SliceLocation`. Files are ordered by filename and then sorted by angular
position derived from `SliceLocation`.

Data locations are supplied through JSON manifests. Relative paths are
resolved from the manifest file's directory. In-vivo splits are lists of case
directories; see `configs/invivo.example.json`. Ex-vivo splits map each case
directory to `[start, stop]` indices used to trim the acquired sweep; see
`configs/exvivo.example.json`.

Create local manifests from the examples and edit their paths before running:

```bash
cp configs/invivo.example.json configs/invivo.json
cp configs/exvivo.example.json configs/exvivo.json
```

## Training

The following architecture and optimization settings reproduce the principal
UltraCCM experiment configuration. Checkpoints are written under `--save_dir`
as `target_modelXXXXXX.pt`.

### In-vivo

```bash
mpiexec -n 1 python -m scripts.train \
  --dataset_mode invivo \
  --data_manifest configs/invivo.json \
  --save_dir outputs/train_invivo \
  --training_mode consistency_training \
  --target_ema_mode adaptive \
  --start_ema 0.95 \
  --scale_mode progressive \
  --start_scales 2 \
  --end_scales 150 \
  --total_training_steps 200000 \
  --loss_norm lpips \
  --kl_loss True \
  --global_batch_size 2 \
  --image_size 256 \
  --lr 0.000005 \
  --num_channels 64 \
  --num_head_channels 64 \
  --num_res_blocks 1 \
  --channel_mult 1,2,4,8,16 \
  --use_scale_shift_norm False \
  --resblock_updown True \
  --use_fp16 True \
  --weight_schedule uniform \
  --epoch 48
```

### Ex-vivo

The paper experiment used self-supervised ex-vivo training. Set
`--paired_supervised True` only when paired dense targets are available; that
mode requires a per-rank batch size of one.

```bash
mpiexec -n 1 python -m scripts.train \
  --dataset_mode exvivo \
  --data_manifest configs/exvivo.json \
  --save_dir outputs/train_exvivo \
  --training_mode consistency_training \
  --target_ema_mode adaptive \
  --start_ema 0.95 \
  --scale_mode progressive \
  --start_scales 2 \
  --end_scales 150 \
  --total_training_steps 200000 \
  --loss_norm lpips \
  --kl_loss True \
  --global_batch_size 2 \
  --image_size 256 \
  --lr 0.000005 \
  --num_channels 64 \
  --num_head_channels 64 \
  --num_res_blocks 1 \
  --channel_mult 1,2,4,8,16 \
  --use_scale_shift_norm False \
  --resblock_updown True \
  --use_fp16 True \
  --weight_schedule uniform \
  --epoch 300
```

## Inference

Run one-step inference with the same architecture used during training:

```bash
mpiexec -n 1 python -m scripts.inference_dicoms \
  --dataset_mode invivo \
  --data_manifest configs/invivo.json \
  --model_path checkpoints/target_model200000.pt \
  --save_dir outputs/invivo \
  --sampler onestep
```

For ex-vivo inference, change `--dataset_mode` and use the ex-vivo manifest and
checkpoint:

```bash
mpiexec -n 1 python -m scripts.inference_dicoms \
  --dataset_mode exvivo \
  --data_manifest configs/exvivo.json \
  --model_path checkpoints/target_model200000.pt \
  --save_dir outputs/exvivo \
  --sampler onestep
```

By default, every fourth axial plane is processed and PNG outputs are saved in
`<save_dir>/<case_id>/{ref,sr,hr}/imgs`. The `hr` output is available only for
ex-vivo data. Use `--slice_stride 1` for every plane or `--max_slices N` for a
short test run.

Add `--save_dicoms True` to write derived DICOM series alongside the PNGs. The
script removes private tags and clears several common identifying fields, but
it is not a certified de-identification tool. Review every generated DICOM for
protected health information before sharing it.


## Reproducibility notes

- The published architecture uses no attention resolutions, so FlashAttention
  is not required for the commands above.
- `axis_distance=15` and `scale=8` are the defaults used by the data pipeline.

## License and acknowledgement

This code is released under the MIT license in `LICENSE`.

UltraCCM builds on the [OpenAI Consistency Models](https://github.com/openai/consistency_models)
implementation. We thank the authors for making their code publicly available.

Please cite UltraCCM when using this code:

```bibtex
@article{pang2026micro,
  title={Micro-ultrasound super-resolution with geometry-driven consistency models},
  author={Pang, Kaifeng and Zhao, Kai and Miao, Qi and Hung, Alex Ling Yu and Oh, Changsuk and Ramirez, Raymi and He, Qiudi and Klein, Jordan and Shao, Wei and Brisbane, Wayne and others},
  journal={Medical Image Analysis},
  pages={104333},
  year={2026},
  publisher={Elsevier}
}
```
