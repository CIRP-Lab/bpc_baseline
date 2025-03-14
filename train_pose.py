import argparse
import torch
from torch.utils.data import DataLoader
from bpc.utils.data_utils import BOPSingleObjDataset, bop_collate_fn
from bpc.pose.models.simple_pose_net import SimplePoseNet
from bpc.pose.models.losses import (
    EulerAnglePoseLoss,
    QuaternionPoseLoss,
    SixDPoseLoss,
)
from bpc.pose.trainers.trainer import train_pose_estimation
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Estimation Model")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Path to dataset root directory (with train_pbr and optionally val)")
    parser.add_argument("--use_real_val", action="store_true",
                        help="If set, use real validation dataset from root_dir/val if available. Otherwise, split train_pbr using train_ratio.")
    parser.add_argument("--target_obj_id", type=int, default=11,
                        help="Target object ID")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="Base directory for checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last checkpoint")
    parser.add_argument("--loss_type", type=str, default="euler", choices=["euler", "quat", "6d"],
                        help="Rotation loss type to use (and set model output dimension accordingly)")
    return parser.parse_args()

def find_scenes(directory):
    """
    Finds scene IDs in a given directory (assuming scene subdirectories are named as digits).
    """
    all_items = os.listdir(directory)
    scene_ids = [item for item in all_items if item.isdigit()]
    scene_ids.sort()
    return scene_ids

def main():
    args = parse_args()

    # Determine training scenes from the "train_pbr" folder.
    train_root = os.path.join(args.root_dir, "train_pbr")
    if not os.path.exists(train_root):
        raise FileNotFoundError(f"Training directory not found: {train_root}")
    train_scene_ids = find_scenes(train_root)
    print(f"[INFO] Found training scene_ids={train_scene_ids}")

    # Determine validation scenes.
    if args.use_real_val:
        real_val_dir = os.path.join(args.root_dir, "val")
        if os.path.exists(real_val_dir):
            val_scene_ids = find_scenes(real_val_dir)
            print(f"[INFO] Found validation scene_ids={val_scene_ids} from {real_val_dir}")
            use_real_val_flag = True
        else:
            print("[WARN] Real validation directory not found, using train split ratio for validation.")
            val_scene_ids = train_scene_ids
            use_real_val_flag = False
    else:
        val_scene_ids = train_scene_ids
        use_real_val_flag = False

    # If using real validation, use the full train_pbr for training.
    train_ratio = 1.0 if use_real_val_flag else 0.8

    obj_id = args.target_obj_id
    checkpoint_dir = os.path.join(args.checkpoints_dir, f"obj_{obj_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create the training dataset with the chosen train_ratio.
    train_ds = BOPSingleObjDataset(
        root_dir=args.root_dir,
        scene_ids=train_scene_ids,
        cam_ids=["cam1", "cam2", "cam3"],
        target_obj_id=obj_id,
        target_size=256,
        augment=False,  # Test
        split="train",
        train_ratio=train_ratio,  # Modified here.
        use_real_val=use_real_val_flag  # Pass the flag if needed.
    )

    # Create the validation dataset.
    val_ds = BOPSingleObjDataset(
        root_dir=args.root_dir,
        scene_ids=val_scene_ids,
        cam_ids=["cam1", "cam2", "cam3"],
        target_obj_id=obj_id,
        target_size=256,
        augment=False,
        split="val",
        use_real_val=use_real_val_flag
    )

    print(f"[INFO] train_ds: {len(train_ds)} samples")
    print(f"[INFO] val_ds: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=bop_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=bop_collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePoseNet(loss_type=args.loss_type, pretrained=not args.resume).to(device)

    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    step_size = max(1, args.epochs // 3)

    # Set the criterion based on loss_type.
    if args.loss_type == "euler":
        criterion = EulerAnglePoseLoss()
    elif args.loss_type == "quat":
        criterion = QuaternionPoseLoss()
    elif args.loss_type == "6d":
        criterion = SixDPoseLoss()
    else:
        raise ValueError("Invalid loss_type")

    criterion_wrapper = criterion

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.8)

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_pose_estimation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  # Use the correct validation loader.
        criterion=criterion_wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        out_dir=checkpoint_dir,
        device=device,
        resume=args.resume
    )

if __name__ == "__main__":
    main()

"""
Example usage:
python3 train_pose.py \
  --root_dir datasets/ \
  --target_obj_id 14 \
  --epochs 10 \
  --batch_size 32 \
  --lr 5e-4 \
  --num_workers 16 \
  --checkpoints_dir bpc/pose/pose_checkpoints/ \
  --loss_type quat \
  --use_real_val
"""
