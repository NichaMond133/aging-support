import tkinter as tk
from tkinter import messagebox
import logging
from ultralytics import YOLO
import cv2
import time
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Check if the script is running as a bundled executable
if getattr(sys, 'frozen', False):
    # If frozen (i.e., running as .exe), the weights folder is in the temp directory
    base_path = sys._MEIPASS
else:
    # If not frozen, use the current working directory (where the script is located)
    base_path = os.path.abspath(".")

# Predefined models and their paths
MODEL_PATHS = {
    "Chin-Tuck": os.path.join(base_path, "yolo-Weights/chin-tuck/chin-tuck-v2-best.pt"),
    "Chin-Out": os.path.join(base_path, "yolo-Weights/forward-head/chin-out-v1.pt"),
    "Hand-to-Head": os.path.join(base_path, "yolo-Weights/hand-to-head/hand-to-head-v1.pt"),
    "Chest-Stretch": os.path.join(base_path, "yolo-Weights/chest-stretch/chest-stretch-v1.pt"),
    "Cervical-Extension": os.path.join(base_path, "yolo-Weights/cervical-extension/cervical-extension-v1.pt"),
}

ACT_MAP = {0: "normal", 1: "exercise"}

# GUI Setup
root = tk.Tk()  # Ensure root window is created first
root.title("YOLO Model Selector")
root.geometry("300x250")

# Variable Initialization after root is created
selected_model_name = tk.StringVar(
    value="Select a Model"
)  # Initialize after root is created


def start_tracking():
    """Initialize YOLO with the selected model and start tracking."""
    global selected_model_name
    model_name = selected_model_name.get()

    if model_name == "Select a Model":
        messagebox.showerror("Error", "Please select a model first.")
        return

    model_path = MODEL_PATHS.get(model_name)
    if not model_path:
        messagebox.showerror("Error", "Invalid model selection.")
        return

    try:
        model = YOLO(model_path)
        logging.info(f"Model '{model_name}' loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")
        return

    # Start video capture and tracking
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not accessible.")
        return

    start_time = time.monotonic()
    prev = None
    exercise_start_time = None  # Time when exercise started
    duration = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.monotonic() - start_time

        cv2.putText(
            frame,
            f"Time: {elapsed_time:.2f} s",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (255, 255, 255),
            2,
        )

        # Run YOLO model on the frame
        results = model.track(frame)
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

        cv2.imshow("YOLO Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Instruction Label
instruction_label = tk.Label(root, text="Select a YOLO model to start tracking:")
instruction_label.pack(pady=10)

# Dropdown Menu for Model Selection
dropdown_menu = tk.OptionMenu(root, selected_model_name, *MODEL_PATHS.keys())
dropdown_menu.pack(pady=10)

# Start Tracking Button
start_tracking_button = tk.Button(root, text="Start Tracking", command=start_tracking)
start_tracking_button.pack(pady=10)

# Exit
exit_label = tk.Label(root, text="(Press 'q' to stop video feed)")
exit_label.pack(pady=10)

tk.Button(root, text="Close Selector", command=root.quit).pack(pady=10)

# Run the GUI loop
root.mainloop()
