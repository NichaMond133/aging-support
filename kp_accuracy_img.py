import collections
import os
import logging
import time
import numpy as np
import json

import cv2
from screeninfo import get_monitors
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_average_differences(json_file="ideal_kp/avg_kp_diff_hand_to_head.json"):
    """
    Load the average keypoint differences from a JSON file.
    """
    try:
        with open(json_file, "r") as f:
            avg_differences = json.load(f)
        return avg_differences
    except Exception as e:
        logging.error(f"Error loading JSON file {json_file}: {e}")
        return {}

def main():

    img_path = "data/exercises/hand-to-head-start.JPG"

    frame = cv2.imread(img_path)

    # Initialize YOLO Model
    try:
        model = YOLO("yolo-Weights/yolo11m-pose.pt")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        return

    # Load the average differences from the JSON file
    avg_differences = load_average_differences()

    # Mapping of indices to keypoint names
    # keypoint_mapping = {10: "right_wrist"}
    # keypoint_mapping = {0: "nose", 2: "right_eye", 4: "right_ear"}
    keypoint_mapping = {2: "right_eye"}

    height, width, _ = frame.shape

    # Track object in frame
    try:
        results = model.track(frame, persist=True)
    except Exception as e:
        logging.error(f"Error tracking objects: {e}")
        return

    if results and results[0].keypoints.xy is not None:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        adjusted_kp = np.copy(keypoints)

        for idx, kp_name in keypoint_mapping.items():
            if kp_name in avg_differences:
                avg_diff = avg_differences[kp_name]
                adjusted_kp[idx, 0] += avg_diff["avg_diff_x"] * width
                adjusted_kp[idx, 1] += avg_diff["avg_diff_y"] * height
                print(f"Adjusted Keypoint {kp_name}: ({adjusted_kp[idx, 0]}, {adjusted_kp[idx, 1]})")

        # Draw adjusted keypoints
        for i, (x, y) in enumerate(adjusted_kp):
            if i in keypoint_mapping:
                color_adjusted = (255, 0, 0)  # Blue for adjusted keypoints
                cv2.circle(frame, (int(x), int(y)), 10, color_adjusted, 2)

    # Preview the frame
    cv2.imshow("Annotated Frame", frame)
    cv2.waitKey(0)

    # Save the annotated frame image
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "ideal_kps/ideal-hand-to-head.jpg"), frame)


if __name__ == "__main__":
    main()