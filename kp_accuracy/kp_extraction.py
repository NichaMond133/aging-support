from ultralytics import YOLO
import cv2
import os
import numpy as np
import json

start_dataset = "data/cervical-extension/starting position"
end_dataset = "data/cervical-extension/end position"

# Initialize the YOLO model
model = YOLO("yolo-Weights/yolo11m-pose.pt")

start_kp = []
end_kp = []

def normalize_keypoints(keypoints, image_width, image_height):
    """
    Normalize the keypoint coordinates to a range between 0 and 1, based on the image dimensions.
    """
    return [(x / image_width, y / image_height) for x, y in keypoints]

def extract_kp(raw_dataset):
    extracted_kp = {}

    # Iterate through images in the folder
    for filename in os.listdir(raw_dataset):
        image_path = os.path.join(raw_dataset, filename)

        # Read the image
        img = cv2.imread(image_path)

        # Get original image size
        img_height, img_width = img.shape[:2]

        # Detect objects
        results = model(source=img, show=False, conf=0.5, stream=False)

        # Annotate and save results
        for i, detection in enumerate(results):
            if detection.boxes:
                keypoints = detection.keypoints.xy if detection.keypoints else None
                if keypoints is not None:
                    # Access keypoints for the single person
                    keypoint = keypoints[0]

                    valid_keypoints = {}

                    # Normalize and store keypoints based on resized image size
                    x_nose = keypoint[0, 0].item()  # Nose x-coordinate
                    y_nose = keypoint[0, 1].item()  # Nose y-coordinate
                    valid_keypoints["nose"] = (x_nose / img_width, y_nose / img_height)

                    x_right_eye = keypoint[2, 0].item()  # Right eye x-coordinate
                    y_right_eye = keypoint[2, 1].item()  # Right eye y-coordinate
                    valid_keypoints["right_eye"] = (x_right_eye / img_width, y_right_eye / img_height)

                    x_right_ear = keypoint[4, 0].item()  # Right ear x-coordinate
                    y_right_ear = keypoint[4, 1].item()  # Right ear y-coordinate
                    valid_keypoints["right_ear"] = (x_right_ear / img_width, y_right_ear / img_height)

                    # x_right_elbow = keypoint[8, 0].item()  # Right elbow x-coordinate
                    # y_right_elbow = keypoint[8, 1].item()  # Right elbow y-coordinate
                    # valid_keypoints["right_elbow"] = (x_right_elbow / img_width, y_right_elbow / img_height)

                    # x_right_wrist = keypoint[10, 0].item()  # Right elbow x-coordinate
                    # y_right_wrist = keypoint[10, 1].item()  # Right elbow y-coordinate
                    # valid_keypoints["right_wrist"] = (x_right_wrist / img_width, y_right_wrist / img_height)


                    extracted_kp[filename] = valid_keypoints
    return extracted_kp

def compute_average_difference(kp_start, kp_end):
    """
    Compute the average difference in keypoints between the start and end dataset.
    This function calculates the average x and y differences for each keypoint (e.g., nose, right_shoulder).
    """
    keypoint_diff = {
        "nose": {"diff_x": 0, "diff_y": 0, "count": 0},
        "right_eye": {"diff_x": 0, "diff_y": 0, "count": 0},
        "right_ear": {"diff_x": 0, "diff_y": 0, "count": 0},
        # "right_wrist": {"diff_x": 0, "diff_y": 0, "count": 0},
    }
    
    # Iterate through all images and compute the difference for each keypoint
    for filename in kp_start:
        if filename in kp_end:
            # Get corresponding keypoints for start and end images
            start_kp = kp_start[filename]
            end_kp = kp_end[filename]
            
            for key in keypoint_diff:
                if key in start_kp and key in end_kp:
                    start_x, start_y = start_kp[key]
                    end_x, end_y = end_kp[key]
                    
                    # Calculate the difference in x and y coordinates
                    diff_x = end_x - start_x  # Horizontal difference (X)
                    diff_y = end_y - start_y  # Vertical difference (Y)
                    
                    # Sum the differences and increment the count
                    keypoint_diff[key]["diff_x"] += diff_x
                    keypoint_diff[key]["diff_y"] += diff_y
                    keypoint_diff[key]["count"] += 1
    
    # Now compute the average for each keypoint
    avg_diff = {}
    for key in keypoint_diff:
        if keypoint_diff[key]["count"] > 0:  # Ensure there are valid keypoints
            avg_diff[key] = {
                "avg_diff_x": keypoint_diff[key]["diff_x"] / keypoint_diff[key]["count"],
                "avg_diff_y": keypoint_diff[key]["diff_y"] / keypoint_diff[key]["count"]
            }

    return avg_diff

kp_start = extract_kp(start_dataset)
kp_end = extract_kp(end_dataset)

average_keypoint_differences = compute_average_difference(kp_start, kp_end)

# Save the average keypoint differences to a JSON file
with open("avg_kp_diff_cervical-extension.json", "w") as f:
    json.dump(average_keypoint_differences, f, indent=4)

print("Saved")