import os
import json
import shutil
import argparse
from tqdm import tqdm
from PIL import Image  # For getting image dimensions

def prepare_train_pbr(train_pbr_path, output_path, obj_id):
    """
    Prepare the train_pbr dataset for YOLO, filtering by specific object ID.
    Creates an 'images' and 'labels' folder under output_path.
    """
    # Cameras to scan. Edit if needed.
    cameras = ["rgb_cam1", "rgb_cam2", "rgb_cam3"]

    # The corresponding ground-truth JSON files in each scene folder
    camera_gt_map = {
        "rgb_cam1": "scene_gt_cam1.json",
        "rgb_cam2": "scene_gt_cam2.json",
        "rgb_cam3": "scene_gt_cam3.json"
    }
    camera_gt_info_map = {
        "rgb_cam1": "scene_gt_info_cam1.json",
        "rgb_cam2": "scene_gt_info_cam2.json",
        "rgb_cam3": "scene_gt_info_cam3.json"
    }

    # Ensure the "images" and "labels" directories exist
    images_dir = os.path.join(output_path, "images")
    labels_dir = os.path.join(output_path, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Iterate over each scene (e.g. 000000, 000001, ...)
    scene_folders = [
        d for d in os.listdir(train_pbr_path)
        if os.path.isdir(os.path.join(train_pbr_path, d)) and not d.startswith(".")
    ]
    scene_folders.sort()  # optional: sort numerically

    for scene_folder in tqdm(scene_folders, desc="Processing train_pbr scenes"):
        scene_path = os.path.join(train_pbr_path, scene_folder)

        # For each camera, read bounding box info
        for cam in cameras:
            rgb_path = os.path.join(scene_path, cam)
            scene_gt_file = os.path.join(scene_path, camera_gt_map[cam])
            scene_gt_info_file = os.path.join(scene_path, camera_gt_info_map[cam])

            if not os.path.exists(rgb_path):
                print(f"Missing RGB folder for {cam} in {scene_folder}: {rgb_path}")
                continue
            if not os.path.exists(scene_gt_file):
                print(f"Missing JSON file for {cam} in {scene_folder}: {scene_gt_file}")
                continue
            if not os.path.exists(scene_gt_info_file):
                print(f"Missing JSON file for {cam} in {scene_folder}: {scene_gt_info_file}")
                continue

            # Load the JSON files for ground truth + info
            with open(scene_gt_file, "r") as f:
                scene_gt_data = json.load(f)
            with open(scene_gt_info_file, "r") as f:
                scene_gt_info_data = json.load(f)

            # Assume image IDs go from 0..N-1
            num_imgs = len(scene_gt_data)  # or use max key from scene_gt_data
            for img_id in range(num_imgs):
                img_key = str(img_id)
                img_file_jpg = os.path.join(rgb_path, f"{img_id:06d}.jpg")
                img_file_png = os.path.join(rgb_path, f"{img_id:06d}.png")
                img_file = img_file_jpg if os.path.exists(img_file_jpg) else img_file_png if os.path.exists(img_file_png) else None

                if img_file is None:
                    continue

                if img_key not in scene_gt_data or img_key not in scene_gt_info_data:
                    # If there's no ground-truth info for this frame, skip
                    continue

                # Filter only bounding boxes for 'obj_id'
                # We also check if visibility fraction > 0 (you can adjust this threshold)
                valid_bboxes = []
                for bbox_info, gt_info in zip(scene_gt_info_data[img_key], scene_gt_data[img_key]):
                    if gt_info["obj_id"] == obj_id and bbox_info["visib_fract"] > 0:
                        valid_bboxes.append(bbox_info["bbox_obj"])  # (x, y, w, h)

                if not valid_bboxes:
                    # No bounding boxes for our object in this image
                    continue

                # Copy the image to the YOLO "images/" folder
                out_img_name = f"{scene_folder}_{cam}_{img_id:06d}.jpg"
                out_img_path = os.path.join(images_dir, out_img_name)
                shutil.copy(img_file, out_img_path)

                # Read real image dimensions
                with Image.open(img_file) as img:
                    img_width, img_height = img.size

                # Write YOLO label(s) for all bounding boxes in this image
                out_label_name = f"{scene_folder}_{cam}_{img_id:06d}.txt"
                out_label_path = os.path.join(labels_dir, out_label_name)
                with open(out_label_path, "w") as lf:
                    for (x, y, w, h) in valid_bboxes:
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width = w / img_width
                        height = h / img_height
                        # YOLO format: class x_center y_center width height
                        # Here class is always '0' because we have only 1 object
                        lf.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def generate_yaml(output_path, obj_id):
    """
    Generate a YOLO .yaml file for training/validation.
    Writes file as: bpc/yolo/configs/data_obj_{obj_id}.yaml
    """
    # Get the directory of THIS script (bpc/yolo/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_configs_dir = os.path.join(script_dir, "configs")
    os.makedirs(yolo_configs_dir, exist_ok=True)

    # The 'images' directory under output_path
    images_dir = os.path.join(output_path, "images")
    train_path = os.path.abspath(images_dir)
    val_path = os.path.abspath(images_dir)

    yaml_path = os.path.join(yolo_configs_dir, f"data_obj_{obj_id}.yaml")

    # YAML content (same as before)
    yaml_content = {
        "train": train_path,
        "val": val_path,
        "nc": 1,
        "names": [f"object_{obj_id}"]
    }

    with open(yaml_path, "w") as f:
        for key, value in yaml_content.items():
            f.write(f"{key}: {value}\n")

    print(f"[INFO] Generated YAML file at: {yaml_path}\n")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare the train_pbr dataset for YOLO training.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the train_pbr dataset (e.g. .../ipd_bop_data_jan25_1/train_pbr).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for YOLO dataset (e.g. .../datasets/yolo11/ipd_bop_data_jan25_1_obj_11).")
    parser.add_argument("--obj_id", type=int, required=True,
                        help="Object ID to filter for (e.g. 11).")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path  = args.output_path
    obj_id       = args.obj_id

    # 1) Prepare YOLO images + labels
    prepare_train_pbr(dataset_path, output_path, obj_id)

    # 2) Generate .yaml file for YOLO
    generate_yaml(output_path, obj_id)

    print("[INFO] Dataset preparation complete!")


if __name__ == "__main__":
    main()