import cv2
import os
from ultralytics import YOLO


def detect_with_yolo(scene_dir, cam_ids, image_id, yolo_model_path):
    """Detect objects using YOLO for all cameras and handle both JPG and PNG formats."""
    yolo = YOLO(yolo_model_path)
    detections = {}

    for cam_id in cam_ids:
        # Try loading JPG or PNG
        img_path_jpg = f"{scene_dir}/rgb_{cam_id}/{image_id:06d}.jpg"
        img_path_png = f"{scene_dir}/rgb_{cam_id}/{image_id:06d}.png"
        
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            detections[cam_id] = []  # No valid image found
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            detections[cam_id] = []
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        yolo_results = yolo(img_rgb)[0]
        det_cam = []

        for box in yolo_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            det_cam.append({"bbox": (x1, y1, x2, y2), "bb_center": (cx, cy)})

        detections[cam_id] = det_cam
    return detections
