# Aging Support

**Source Code Location:** `exercise_detection/exercise_detection_gui.py`.

This program uses a custom-trained pose estimation model based on the YOLO (You Only Look Once) to detect and monitor neck exercises specifically designed for the elderly. The application provides real-time feedback by tracking and timing the exercises through a user-friendly graphical interface built with Tkinter.

---

## Overview

The Aging Support Exercise Detection GUI is designed to:
- **Monitor Neck Exercises:** Recognize and time different neck exercises such as chin-tuck, chin-out, hand-to-head, chest-stretch, and cervical extension.
- **Provide Real-Time Feedback:** Overlay tracking information such as elapsed time, current activity state (normal or exercise), and exercise duration directly on the video feed.
- **User-Friendly Interface:** Allow users to select a model through a simple Tkinter GUI, making it accessible even for non-technical users.

The program leverages a custom-trained YOLO model to detect neck exercises. It supports multiple exercise types by loading the appropriate model weights from the local `yolo-Weights` directory.

---

## Features

- **Custom Model Selection:** Choose from several pre-trained models for different neck exercises.
- **Real-Time Video Processing:** Uses OpenCV to capture live video feed and display annotated frames.
- **Activity Tracking:** Displays the elapsed time, current activity status (`normal` or `exercise`), and duration of the exercise phase.
- **User Notifications:** Error messages and instructions are provided via the GUI using Tkinter message boxes.
- **Cross-Platform Support:** Compatible with both bundled executable environments (e.g., using PyInstaller) and standard Python script execution.

---
