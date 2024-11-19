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

def load_average_differences(json_file="avg_kp_diff.json"):
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
    video_path = "videos/chin-tuck.mp4"

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

    track_hist = collections.defaultdict(list)
    curr = None

    MAX_TRACK_HISTORY = 30

    # Load the average differences from the JSON file
    avg_differences = load_average_differences()

    while cap.isOpened():
        cap.set(3, 640)
        cap.set(4, 480)

        success, frame = cap.read()

        if not success:
            logging.info("End of video stream.")
            break

        frame_count += 1
        elapsed_time = frame_count / frame_rate

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

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box

                # Get the keypoints for the current detection
                keypoints = results[0].keypoints.xy if results[0].keypoints else None
                if keypoints is not None:
                    keypoint = keypoints[0].clone()

                    if track_id not in track_hist:
                        track_hist[track_id] = {'adjusted_keypoints': None} 

                    if track_hist[track_id]['adjusted_keypoints'] is None:
                        if keypoint is not None and avg_differences:
                            if 'nose' in avg_differences:
                                avg_diff_nose = avg_differences["nose"]
                                keypoint[0, 0] += avg_diff_nose["avg_diff_x"]
                                keypoint[0, 1] += avg_diff_nose["avg_diff_y"]
                            
                            if 'right_eye' in avg_differences:
                                avg_diff_eye = avg_differences["right_eye"]
                                keypoint[6, 0] += avg_diff_eye["avg_diff_x"]
                                keypoint[6, 1] += avg_diff_eye["avg_diff_y"]
                            
                            if 'right_ear' in avg_differences:
                                avg_diff_ear = avg_differences["right_ear"]
                                keypoint[4, 0] += avg_diff_ear["avg_diff_x"]
                                keypoint[4, 1] += avg_diff_ear["avg_diff_y"]
                            
                            track_hist[track_id]['adjusted_keypoints'] = keypoint
                            
                            cv2.putText(
                            frame,
                            f"Calibrating...",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.25,
                            (255, 255, 255),
                            2,
                            )
                    
                    adjusted_keypoints = track_hist[track_id]['adjusted_keypoints']

                    # Draw the adjusted keypoints on the frame
                    for i, (x, y) in enumerate(adjusted_keypoints):
                        if i == 0:  # nose
                            color = (0, 255, 0)
                        elif i == 6:  # right_shoulder
                            color = (0, 0, 255)
                        elif i == 4:  # right_ear
                            color = (255, 0, 0)
                        else:
                            color = (255, 255, 255)

                        cv2.circle(frame, (int(x), int(y)), 10, color, -5)

        # Annotate and display the frame
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()