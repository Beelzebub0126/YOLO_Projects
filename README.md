# YOLO_Projects

# Object_Detection_and_Instance_Segmentation_of_PHP_Bills

This repository contains two Python scripts for real-time detection and segmentation of Philippine Peso (PHP) banknotes using custom YOLO models trained only on the main denominations: 1, 5, 10, 20, 50, 100, 200, 500, and 1000 PHP bills.[file:3][file:4] The scripts process webcam input for live inference, displaying bounding boxes for detection and masks for segmentation, with FPS metrics for performance evaluation.[file:3][file:4]

## Features

- Real-time object detection and instance segmentation focused exclusively on specified PHP bills, ignoring other objects.[file:3][file:4]
- Interactive modes in segmentation script for comparing classical techniques (thresholding, edge detection) with YOLO.[file:3]
- Webcam-based testing with confidence thresholds and visual overlays for accuracy assessment.[file:3][file:4]

## Requirements

- Python 3.8 or higher.[file:3][file:4]
- Create a virtual environment: `python -m venv env`.[file:3][file:4]
- Activate it: `source env/bin/activate` (macOS/Linux) or `env\Scripts\activate` (Windows).[file:3][file:4]
- Install libraries: `pip install ultralytics opencv-python numpy` (includes PyTorch via Ultralytics).[file:3][file:4]
- Place model files (Object_Detection_Model.pt and Instance_Segmentation_Model.pt) in the repository root.[file:3][file:4]

## Usage

1. Ensure webcam access and good lighting for testing PHP bills.[file:3][file:4]
2. Run detection: `python Object_Detection.py` (press 'q' to quit).[file:4]
3. Run segmentation: `python Instance_Segmentation.py` (toggle modes with 't', 'e', 's'; 'q' to quit).[file:3]
4. Models detect only the nine PHP denominations; retrain for expansions.[web:5][web:9]

## Scripts

### Object_Detection.py
Loads YOLO detection model for bounding boxes, labels, and FPS on webcam feed of PHP bills.[file:4]

### Instance_Segmentation.py
Implements multi-mode segmentation: Otsu thresholding ('t'), Canny edges ('e'), YOLO masks ('s') with labels and average FPS.[file:3]

For issues or contributions, open a GitHub issue.[web:14][web:23]
