"""
Step 5 of Project 1: test your trained YOLO model live, next to AprilTag.

Shows both detectors on the same camera image so you can compare them:
  - Green boxes  = YOLO found the landing pad (with confidence %)
  - Blue outline = AprilTag found the tag
The top lines show whether each one detected something, and how long it took.

Video source: by default the ZED on the Jetson. First start the stream on the
Jetson (cd ~/sultan && python3 stream_capture.py), keep its "Detect AprilTag"
button OFF, and close the browser page so it doesn't compete for the stream.

Run (in the nvidia-project folder):   python detect_yolo.py
Quit: press q in the video window.   Save a screenshot: press s
"""

import time

import cv2
import torch
from ultralytics import YOLO

# ---- Settings you may need to change ----------------------------------------
MODEL_PATH = "runs/detect/pad_v1/weights/best.pt"
# Where the video comes from:
#   "http://192.168.55.1:8000/stream" = ZED on the Jetson (run stream_capture.py there)
#   0, 1, 2 ...                       = a camera plugged into this Mac
CAMERA_INDEX = "http://192.168.55.1:8000/stream"
USE_ZED = True     # True = cut a side-by-side ZED image to the left lens (if needed)
CONF = 0.5         # only show YOLO detections at least this confident (0-1)
# -----------------------------------------------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO(MODEL_PATH)
print(f"Loaded {MODEL_PATH}, running on {device}")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
tag_detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {CAMERA_INDEX}. Try another CAMERA_INDEX.")
if USE_ZED and isinstance(CAMERA_INDEX, int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps = 0.0
prev = time.time()
shot = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read a frame from the camera.")
        break
    if USE_ZED and frame.shape[1] >= 2 * frame.shape[0]:
        frame = frame[:, : frame.shape[1] // 2]

    # --- YOLO ---
    result = model.predict(frame, imgsz=640, conf=CONF, device=device, verbose=False)[0]
    yolo_ms = result.speed["inference"]
    yolo_best = 0.0
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        yolo_best = max(yolo_best, conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"pad {conf:.0%}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- AprilTag ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    corners, ids, _ = tag_detector.detectMarkers(gray)
    tag_ms = (time.perf_counter() - t0) * 1000
    tag_found = ids is not None
    if tag_found:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(255, 128, 0))

    # --- Status text ---
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
    prev = now
    yolo_txt = f"YOLO: {'FOUND ' + format(yolo_best, '.0%') if yolo_best else 'no pad'}  ({yolo_ms:.1f} ms)"
    tag_txt = f"AprilTag: {'FOUND' if tag_found else 'no tag'}  ({tag_ms:.1f} ms)"
    for i, (txt, color) in enumerate([(yolo_txt, (0, 255, 0)),
                                      (tag_txt, (255, 160, 0)),
                                      (f"FPS: {fps:.1f}", (255, 255, 255))]):
        cv2.putText(frame, txt, (10, 30 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, txt, (10, 30 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("YOLO vs AprilTag (q = quit, s = screenshot)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        name = f"screenshot_{shot:02d}.jpg"
        cv2.imwrite(name, frame)
        print(f"Saved {name}")
        shot += 1

cap.release()
cv2.destroyAllWindows()
