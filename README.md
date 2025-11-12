# YOLO_Projects

# Object_Detection_and_Instance_Segmentation_of_PHP_Bills

- Python project for real-time detection and segmentation of PHP banknotes (1, 5, 10, 20, 50, 100, 200, 500, 1000 denomination bills) using YOLO models.

## Folder Structure

- Object_Detection_and_Instance_Segmentation_of_PHP_Bills
    - Codes
        - Object_Detection.py      # detection script
        - Instance_Segmentation.py # segmentation script
    - Models
        - Object_Detection_Model.pt        # detection model
        - Instance_Segmentation_Model.pt   # segmentation model

## Features

- Detects and segments only main PHP bills from webcam.
- Instance segmentation script has modes:
    - 't': thresholding
    - 'e': edge detection
    - 's': YOLO segmentation (default)

## Setup (VSCode)

- Python 3.8+ required
- Open terminal, create and activate venv:
    python -m venv env
    env\Scripts\activate
- Install libraries:
    pip install ultralytics opencv-python numpy

## Usage

- cd Codes
- Run detection:
    python Object_Detection.py
- Run segmentation:
    python Instance_Segmentation.py

- Models only detect specified PHP bills.

