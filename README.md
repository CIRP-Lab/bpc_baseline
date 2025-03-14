# Object Detection and Pose Estimation Pipeline

## Overview
This repository provides an end-to-end pipeline for:
- Preparing data for YOLO-based object detection using Ultralytics’ YOLO.
- Training a YOLO model to detect a specific object.
- Training a simple pose estimation model (SimplePoseNet) on BOP-format data.

## Repository Structure
```
bpc/
  yolo/
    prepare_data.py   # Prepare data in YOLO format
    train.py          # Train YOLO using Ultralytics
    configs/         # YOLO configuration files (.yaml)
    models/          # YOLO trained weights (.pt)

  pose/
    checkpoints/     # Stores pose model checkpoints
    models/          # Network definitions, losses, etc.
    trainers/        # Training logic for pose estimation

  utils/
    data_utils.py    # BOP dataset loading + transforms

  inference/
    # Scripts for epipolar matching, YOLO detection, and match pipelines

train_pose.py       # Pose estimation training script
datasets/           # Contains BOP and YOLO data
runs/               # YOLO training runs, logs, etc.
```

## Environment Setup
To set up the environment, follow these steps (tested on Ubuntu with an NVIDIA GPU). The environment name is `bop`.

### Build Docker Container
```bash
cd docker/
docker build . -t bpc:2025.1.31
```

### Run Docker
```bash
docker run -p 8888:8888 --shm-size=1g --runtime nvidia --gpus all -v $(pwd):/code -ti bpc:2025.1.31 bash
cd /code
```
**Hint:** Ensure you return to the `bpc_baseline` folder before running the pipeline.

### Download Data
```bash
bash download_data.sh
```
**Note:** If you plan to download the dataset and train your model, remember that for submission you should delete the downloaded datasets in the `datasets` folder. This prevents the Docker image from growing too large.

## Training Pipeline

### Prepare YOLO Data
Convert BOP data to YOLO format:
```bash
python3 bpc/yolo/prepare_data.py \
    --dataset_path "datasets/ipd/train_pbr" \
    --output_path "datasets/yolo11/train_obj_8" \
    --obj_id 8
```
**Hint:** You can modify the visibility threshold (`visib_fract`) in `prepare_data.py`.

### Train YOLO Model
```bash
python3 bpc/yolo/train.py \
    --obj_id 8 \
    --data_path "bpc/yolo/configs/data_obj_8.yaml" \
    --epochs 20 \
    --imgsz 1280 \
    --batch 16 \
    --task detection
```
*Updated `--imgsz` from 640 to 1280.*

### Train Pose Model
```bash
python3 train_pose.py \
  --root_dir datasets/ipd \
  --target_obj_id 8 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --num_workers 8 \
  --checkpoints_dir bpc/pose/pose_checkpoints/ \
  --loss_type quat
```
**Hints:**
- Use `--use_real_val` to utilize the `val` folder in `datasets/ipd/` as the validation set. Without this flag, the script will split the `train_pbr` data in an 80:20 ratio.
- Use `--resume` to continue training from the last checkpoint if available.

### Download Pretrained Models
```bash
wget https://storage.googleapis.com/akasha-public/IBPC/baseline_solution/v1/models.zip
unzip models.zip
rm models.zip
```

### Run Inference
```bash
jupyter notebook --ip=0.0.0.0 --allow-root --port=8888
# Go to localhost:8888 on your browser
# Run "Inference Notebook.ipynb"
```

### Notes
- Ensure CUDA 12.1 drivers are installed and PyTorch recognizes the GPU (`nvidia-smi`).
- BOP dataset must follow standard conventions (train_pbr, test, etc.).
- Update `bpc/yolo/configs/data_obj_8.yaml` with the correct dataset paths.
- If encountering module import errors, try:
  ```bash
  python -m idp_codebase.pose.train ...
  ```
  or add `__init__.py` files where necessary.

## Detailed Documentation
For detailed documentation, please refer to [this page](blog/documentation.md).