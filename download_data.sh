#!/bin/bash

set -e  # Exit if a command fails

### 1️⃣ Install Dependencies (Aria2 & 7z) ###
install_aria2() {
    if ! command -v aria2c &> /dev/null; then
        echo "[INFO] Aria2 is not installed. Installing now..."
        sudo apt update && sudo apt install -y aria2
    fi
}

install_7z() {
    if ! command -v 7z &> /dev/null; then
        echo "[INFO] 7z is not installed. Installing now..."
        sudo apt update && sudo apt install -y p7zip-full
    fi
}

install_aria2
install_7z

### 2️⃣ Define Dataset Info ###
export SRC="https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main"
DATASET_FILES=(
    "ipd_base.zip"
    "ipd_models.zip"
    "ipd_val.zip"
    "ipd_test_all.zip"
    "ipd_test_all.z01"
    "ipd_train_pbr.zip"
    "ipd_train_pbr.z01"
    "ipd_train_pbr.z02"
    "ipd_train_pbr.z03"
)

TARGET_DIR="./datasets"

# Ensure dataset directory exists
mkdir -p "$TARGET_DIR"

### 3️⃣ Fast Download Function (Aria2) ###
download_file() {
    local file="$1"
    local url="$SRC/$file"
    local filepath="$TARGET_DIR/$file"

    if [ -f "$filepath" ]; then
        echo "[INFO] File $filepath already exists. Skipping download."
        return 0
    fi

    echo "[INFO] Downloading $file with Aria2..."
    aria2c -x 16 -s 16 -k 1M -d "$TARGET_DIR" -o "$file" "$url"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Download failed for $file. Check your connection."
        exit 1
    fi
}

# Download all dataset files
for file in "${DATASET_FILES[@]}"; do
    download_file "$file"
done

### 4️⃣ Extract Files Exactly Like Original Commands ###
cd "$TARGET_DIR"

echo "[INFO] Extracting dataset as per original instructions..."

7z x ipd_base.zip             # Contains folder "ipd".
7z x ipd_models.zip -oipd     # Unpacks to "ipd".
7z x ipd_val.zip -oipd        # Unpacks to "ipd".
7z x ipd_test_all.zip -oipd   # Unpacks to "ipd".
7z x ipd_train_pbr.zip -oipd  # Unpacks to "ipd".

### 5️⃣ Cleanup (Remove Zip Files After Successful Extraction) ###
rm -f ipd_base.zip ipd_models.zip ipd_val.zip
rm -f ipd_test_all.zip ipd_test_all.z01
rm -f ipd_train_pbr.zip ipd_train_pbr.z01 ipd_train_pbr.z02 ipd_train_pbr.z03

echo "[INFO] Dataset fully downloaded, extracted, and cleaned up."