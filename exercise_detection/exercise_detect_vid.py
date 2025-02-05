import collections
import os
import logging
import time 
import numpy as np

import cv2
from screeninfo import get_monitors
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
        
    video_path = "videos/cervical-extension.mp4"

    # Initialize YOLO Model
    try:
        model = YOLO("yolo-Weights/cervical-extension/cervical-extension-v1.pt")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return

    monitors = get_monitors()
    if not monitors:
        logging.error("No monitors found.")
        return

    screen_width, screen_height = monitors[0].width, monitors[0].height
    frame_rate, frame_count, elapsed_time = cap.get(cv2.CAP_PROP_FPS), 0, 0

    # track_hist = collections.defaultdict(list)
    start_time = time.time()
    curr = None
    prev = None
    in_exercise = False  # Flag to track if the person is in exercise
    exercise_start_time = None  # Time when exercise started
    duration = None


    # Activity classes map
    act_map = {0: "normal", 1: "exercise"}
    # MAX_TRACK_HISTORY = 30

    # Activity log dictionary
    # act_dict = {
    #     "prev": None,
    #     "exercise": {"start_time": None, "duration": 0},
    #     "normal": {"start_time": None, "duration": 0},
    # }

    while cap.isOpened():

        cap.set(3, screen_width)
        cap.set(4, screen_height)

        success, frame = cap.read()

        if not success:
            logging.info("End of video stream.")
            break

        frame = cv2.resize(frame, (screen_width, screen_height))
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
            activity = results[0].boxes.cls.cpu().numpy().astype(int)[0]

            if activity == 0 and prev != 0:
                exercise_start_time = elapsed_time
                
            if activity == 1 and prev == 0:
                exercise_end_time = elapsed_time
                duration = round(exercise_end_time - exercise_start_time, 2)
            
            prev = activity

            cv2.putText(
                    frame,
                    f"Current Act: {act_map.get(activity)}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.25,
                    (255, 255, 255),
                    2,
                )
            
            cv2.putText(
                    frame,
                    f"duration: {duration}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.25,
                    (255, 255, 255),
                    2,
                )

            # for box, track_id in zip(boxes, track_ids):
            #     x, y, w, h = box
                # track = track_hist[track_id]
                # track.append((float(x), float(y)))

                # Determine current activity
                # curr = act_map.get(activity)

                # if len(track) >= 10:
                #     x_diff = track[-1][0] - track[-10][0]
                #     cv2.putText(
                #     frame,
                #     f"X Diff: {x_diff}",
                #     (10, 200),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1.0,
                #     (0, 0, 0),
                #     2,
                # )

                # Update start time if activity just started
                # if act_dict[curr]["start_time"] is None:
                #     act_dict[curr]["start_time"] = round(elapsed_time, 2)

                # i = 0
                # for key, value in act_dict.items():
                #     if key != "prev":
                #         cv2.putText(
                #             frame,
                #             f"{key}: {value['duration']} s",
                #             (10, 120 + (i * 35)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             1.0,
                #             (0, 0, 0),
                #             2,
                #         )
                #         i += 1


            # if len(track) > MAX_TRACK_HISTORY:
            #     track.pop(0)

        # Update duration if activity hasn't changed
        # if (
        #     act_dict["prev"] is not None
        #     and act_dict[curr]["start_time"] is not None
        #     and act_dict["prev"] == curr
        # ):
        #     act_dict[curr]["duration"] = round(
        #         elapsed_time - act_dict[curr]["start_time"], 2
        #     )

        # act_dict["prev"] = curr

        annotated_frame = results[0].plot()
        # Draw the tracking lines
        # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        cv2.imshow("YOLOv11 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
