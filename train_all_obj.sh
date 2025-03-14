#!/bin/bash

# Define the dataset path and training parameters
ROOT_DIR="datasets/ipd"
EPOCHS=105  # Target number of epochs
BATCH_SIZE=32
LR=1e-4
NUM_WORKERS=16
CHECKPOINTS_DIR="bpc/pose/pose_checkpoints/"
LOSS_TYPE="quat"

# List of object IDs to train
OBJ_IDS=(0 1 4 8 10 11 14 18 19 20)

# Loop through each object ID
for OBJ_ID in "${OBJ_IDS[@]}"
do
    # Define the checkpoint path for this object
    CHECKPOINT_PATH="${CHECKPOINTS_DIR}/obj_${OBJ_ID}/last_checkpoint.pth"

    # Assume training is needed
    TRAIN_NEEDED=true

    # Check if the checkpoint exists
    if [ -f "$CHECKPOINT_PATH" ]; then
        # Read the last saved epoch from the checkpoint
        LAST_EPOCH=$(python3 -c "import torch; print(torch.load('${CHECKPOINT_PATH}')['epoch'])" 2>/dev/null)

        # If training already reached the target epochs, skip it
        if [ "$LAST_EPOCH" -ge "$EPOCHS" ]; then
            echo "[SKIP] Training for obj_id ${OBJ_ID} is already completed (Epoch $LAST_EPOCH / $EPOCHS)."
            TRAIN_NEEDED=false
        fi
    fi

    # If training is needed, run the command
    if [ "$TRAIN_NEEDED" = true ]; then
        echo "[START] Training obj_id ${OBJ_ID}..."
        
        # Conditionally add --resume flag
        RESUME_FLAG=""
        if [ -f "$CHECKPOINT_PATH" ]; then
            RESUME_FLAG="--resume"
        fi

        python3 train_pose.py \
            --root_dir "$ROOT_DIR" \
            --target_obj_id "$OBJ_ID" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --num_workers "$NUM_WORKERS" \
            --checkpoints_dir "$CHECKPOINTS_DIR" \
            --loss_type "$LOSS_TYPE" \
            $RESUME_FLAG  # Conditionally added
    fi
done

echo "All training tasks completed!"