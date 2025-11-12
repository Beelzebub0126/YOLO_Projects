# test_yolo11n.py - Test Classes and Detection for yolo11n.pt

from ultralytics import YOLO
import cv2
import time

# Load the yolo11n.pt model (pre-trained on COCO)
model_path = r"Object_Detection_Model.pt"
model = YOLO(model_path, task="detect")

# Print all classes to console (should show 80 COCO classes)
print("Model loaded successfully.")
print(f"Number of classes: {len(model.names)}")
print(f"Class names (first 10): {dict(list(model.names.items())[:10])}")  # Sample
print("Full classes: ", model.names)  # Full list

# Open webcam for real-time testing
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit. Point camera at objects to test detection (e.g., person, car).")

conf_thresh = 0.5  # Confidence threshold

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    t0 = time.perf_counter()

    # Run inference
    results = model(frame, conf=conf_thresh, verbose=False)
    annotated_frame = results[0].plot() if results[0].boxes is not None else frame.copy()

    t1 = time.perf_counter()
    fps = 1.0 / max(t1 - t0, 1e-6)

    # Overlay FPS
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display frame
    cv2.imshow("yolo11n Detection Test", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test complete. If detections are generic (not PHP bills), use your custom trained model instead.")
