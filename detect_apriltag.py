"""
Step 1 of Project 1: live AprilTag detection from your Mac camera.

What it does:
  - Opens your webcam
  - Finds AprilTags (family 36h11) in every frame
  - Draws a box + ID on each tag and a dot at its center
  - Shows FPS and how many milliseconds detection took

Run:   python detect_apriltag.py
Quit:  press q in the video window
"""

import time
import cv2

# 1. Set up the detector. 36h11 is the most common AprilTag family.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

# 2. Open the camera. 0 = the built-in camera.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the camera. Check System Settings > Privacy & Security > Camera.")
    raise SystemExit(1)

fps = 0.0
prev_time = time.time()

while True:
    # 3. Grab one frame from the camera.
    ok, frame = cap.read()
    if not ok:
        print("Could not read a frame from the camera.")
        break

    # 4. Detection works on grayscale images.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Detect tags and time how long it takes.
    t0 = time.perf_counter()
    corners, ids, _ = detector.detectMarkers(gray)
    detect_ms = (time.perf_counter() - t0) * 1000

    # 6. Draw results.
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            cx, cy = tag_corners[0].mean(axis=0).astype(int)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            print(f"Tag {tag_id} at pixel ({cx}, {cy})")

    # 7. Smoothed frames-per-second.
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}   detect: {detect_ms:.1f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("AprilTag detector (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
