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

def calculate_accuracy(actual_kp, adjusted_kp):
    """
    Calculate accuracy based on the Euclidean distance between
    actual and adjusted keypoints.
    """
    if actual_kp is None or adjusted_kp is None:
        return 0.0

    distance = np.linalg.norm(actual_kp - adjusted_kp)
    #chin-out 150, chin-in 80, chest-stretch 500
    max_distance = 150  # Threshold for maximum reasonable deviation
    accuracy = max(0, 100 - (distance / max_distance) * 100)
    return accuracy

def main():
    video_path = "videos/hand-to-head.mp4"

    # Initialize YOLO Model
    try:
        model = YOLO("yolo-Weights/yolo11m-pose.pt")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return

    frame_rate, frame_count, elapsed_time = cap.get(cv2.CAP_PROP_FPS), 0, 0

    cap.set(3, 640)  # width
    cap.set(4, 480)  # height

    # Load the average differences from the JSON file
    avg_differences = load_average_differences()

    # Mapping of indices to keypoint names
    # keypoint_mapping = {0: "nose", 2: "right_eye", 4: "right_ear"}
    # keypoint_mapping = {10: "right_wrist"}
    keypoint_mapping = {2: "right_eye"}

    # Record the start time
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            logging.info("End of video stream.")
            break

        frame_count += 1
        elapsed_time = frame_count / frame_rate
        height, width, _ = frame.shape

        # Display information on frame
        cv2.putText(
            frame,
            f"Time: {elapsed_time:.2f} s",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (255, 255, 255),
            2,
        )

        # Track object in frame
        try:
            results = model.track(frame, persist=True)
        except Exception as e:
            logging.error(f"Error tracking objects: {e}")
            continue

        if results and results[0].keypoints.xy is not None:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()

            if time.time() - start_time <= 5:
                # Adjust keypoints using average differences
                adjusted_kp = np.copy(keypoints)
                for idx, kp_name in keypoint_mapping.items():
                    if kp_name in avg_differences:
                        avg_diff = avg_differences[kp_name]
                        adjusted_kp[idx, 0] += avg_diff["avg_diff_x"] * width
                        adjusted_kp[idx, 1] += avg_diff["avg_diff_y"] * height
                
                cv2.putText(
                            frame,
                            f"Calibrating...",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.00,
                            (255, 255, 255),
                            2,
                            )
            else:
                # Calculate accuracy
                # for idx in keypoint_mapping.keys():
                #     #change to idx for hand-to-head
                #     actual_kp = keypoints[idx]
                #     adj_kp = adjusted_kp[idx]
                #     accuracy = calculate_accuracy(actual_kp, adj_kp)

                accuracy = calculate_accuracy(keypoints[9], adjusted_kp[2])

                cv2.putText(
                    frame,
                    f"Accuracy: {accuracy:.2f}%",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.00,
                    (255, 255, 255),
                    2,
                )

            # Draw keypoints and adjusted keypoints
            for i, (x, y) in enumerate(adjusted_kp):
                if i in keypoint_mapping: 
                    color_adjusted = (255, 0, 0)
                    cv2.circle(frame, (int(x), int(y)), 5, color_adjusted, 2)

        # Annotate and display the frame
        annotated_frame = results[0].plot()
        cv2.imshow("Exercise Accuracy", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()