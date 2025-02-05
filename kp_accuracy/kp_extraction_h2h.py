from ultralytics import YOLO
import cv2
import os
import numpy as np
import json

# Path to dataset and YOLO weights
end_dataset = "data/hand-to-head/end position"
model = YOLO("yolo-Weights/yolo11m-pose.pt")

def extract_kp(raw_dataset):
    extracted_kp = {}
    for filename in os.listdir(raw_dataset):
        image_path = os.path.join(raw_dataset, filename)

        img = cv2.imread(image_path)
        if img is None:
            continue

        img_height, img_width = img.shape[:2]

        results = model(source=img, conf=0.5, stream=False)
        for detection in results:
            if detection.boxes and detection.keypoints:
                keypoint = detection.keypoints.xy[0]  # Assuming single person per image
                valid_keypoints = {
                    "right_eye": (keypoint[2, 0].item() / img_width, keypoint[2, 1].item() / img_height),
                    "right_wrist": (keypoint[10, 0].item() / img_width, keypoint[10, 1].item() / img_height)
                }
                extracted_kp[filename] = valid_keypoints
    return extracted_kp

def compute_average_difference(kp_end):
    total_x_diff, total_y_diff = 0, 0
    count = 0
    for filename in kp_end:
        if "right_eye" in kp_end[filename] and "right_wrist" in kp_end[filename]:
            x_eye, y_eye = kp_end[filename]["right_eye"]
            x_wrist, y_wrist = kp_end[filename]["right_wrist"]
            total_x_diff += abs(x_eye - x_wrist)
            total_y_diff += abs(y_eye - y_wrist)
            count += 1
    if count > 0:
        return {"avg_x_diff": total_x_diff / count, "avg_y_diff": total_y_diff / count}
    return {"avg_x_diff": 0, "avg_y_diff": 0}

# Extract and compute
kp_end = extract_kp(end_dataset)
average_keypoint_differences = compute_average_difference(kp_end)

# Save result
output_file = "avg_kp_diff_hand-to-head.json"
with open(output_file, "w") as f:
    json.dump(average_keypoint_differences, f, indent=4)

print(f"Saved to {output_file}")
