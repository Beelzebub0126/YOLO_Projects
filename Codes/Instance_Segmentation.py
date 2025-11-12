# Ultralytics.py - Modified for Instance Segmentation Techniques with YOLO (.pt model) and FPS

from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load the YOLO instance segmentation model (replace with your .pt path)
model = YOLO("Instance_Segmentation_Model.pt", task="segment")

# Open webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit, 't' for thresholding, 'e' for edge-based, 's' for instance segmentation.")

def apply_thresholding(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return thresh_colored

def apply_edge_based(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_frame = frame.copy()
    cv2.drawContours(edge_frame, contours, -1, (0, 255, 0), 2)
    return edge_frame

def apply_instance_segmentation(frame):
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    if results[0].masks is not None and results[0].masks.data is not None:
        masks = results[0].masks.data.cpu().numpy()  # Shape: (num_masks, H, W)
        boxes = results[0].boxes  # For class and conf, but not drawing boxes
        for i in range(len(masks)):
            mask = masks[i] > 0.5  # Binary mask threshold
            # Get class and confidence
            class_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            class_name = model.names[class_id] if class_id in model.names else f'Class {class_id}'
            # Color for mask (cycle through colors for instances)
            color = np.random.randint(0, 255, size=3).tolist()
            # Overlay mask
            annotated_frame[mask] = annotated_frame[mask] * 0.3 + np.array(color) * 0.7
            # Add class and conf text near the mask centroid
            mask_y, mask_x = np.where(mask)
            if len(mask_x) > 0 and len(mask_y) > 0:
                centroid_x = int(np.mean(mask_x))
                centroid_y = int(np.mean(mask_y))
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(annotated_frame, label, (centroid_x, centroid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated_frame

mode = 's'  # Default to instance segmentation

# FPS calculation buffer
frame_rate_buffer = []
fps_avg_len = 50
avg_fps = 0.0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    t0 = time.perf_counter()

    if mode == 't':
        annotated_frame = apply_thresholding(frame)
        cv2.putText(annotated_frame, "Thresholding Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    elif mode == 'e':
        annotated_frame = apply_edge_based(frame)
        cv2.putText(annotated_frame, "Edge-Based Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        annotated_frame = apply_instance_segmentation(frame)
        cv2.putText(annotated_frame, "Instance Segmentation Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    t1 = time.perf_counter()
    inst_fps = 1.0 / max(t1 - t0, 1e-6)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(inst_fps)
    avg_fps = np.mean(frame_rate_buffer) if frame_rate_buffer else inst_fps

    # Overlay FPS
    cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Segmentation Techniques", annotated_frame)

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        mode = 't'
    elif key == ord('e'):
        mode = 'e'
    elif key == ord('s'):
        mode = 's'

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
print(f"Average FPS over session: {avg_fps:.2f}")
