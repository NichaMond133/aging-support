import collections
import time
import logging
import torch

import cv2
from screeninfo import get_monitors
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():

    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Initialize YOLO Model
    try:
        model = YOLO("yolo-Weights/chest-stretch/chest-stretch-v1.pt")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        return
    
    cap = None

    try:
        cap = cv2.VideoCapture(0)

        if cap is None:
            logging.error("No camera available. Exiting...")
            return

        cap.set(3, 640)  # width
        cap.set(4, 480)  # height

        track_hist = collections.defaultdict(list)
        MAX_TRACK_HISTORY = 30
        start_time = time.monotonic()
        curr = None
        prev = None
        in_exercise = False  # Flag to track if the person is in exercise
        exercise_start_time = None  # Time when exercise started
        duration = None

        # Activity classes map
        ACT_MAP = {0: "normal", 1: "exercise"}


        while cap.isOpened():
            success, frame = cap.read()

            # Continuously check for working camera
            if not success:
                logging.warning("Camera feed lost, attempting to switch to backup...")
                if cap is None:
                    logging.error("No camera available. Exiting...")
                    break

            elapsed_time = time.monotonic() - start_time

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
            results = model.track(frame, persist=True)
            if results and results[0].boxes.id is not None:
                activity = results[0].boxes.cls.cpu().numpy().astype(int)[0]

                if activity == 0 and prev != 0:
                    exercise_start_time = time.monotonic()
                
                if activity == 1 and prev == 0:
                    exercise_end_time = time.monotonic()
                    duration = round(exercise_end_time - exercise_start_time, 2)
                    
                prev = activity

                cv2.putText(
                    frame,
                    f"Current Act: {ACT_MAP.get(activity, "None")}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.25,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Duration: {duration}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.25,
                    (255, 255, 255),
                    2,
                )

            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        logging.error(f"Error during video processing: {e}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
