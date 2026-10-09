"""
Step 2 of Project 1: collect training photos of your landing pad.

Works with your Mac's built-in camera OR a ZED camera plugged into the Mac.

Controls (click the video window first):
  SPACE  save one photo
  a      turn auto-save on/off (saves a photo every 0.5 s while you move)
  q      quit

Photos are saved to: dataset/images/
"""

import os
import time
import cv2

# ---- Settings you may need to change ----------------------------------------
CAMERA_INDEX = 1     # 0 = usually the Mac's built-in camera, 1 = usually the ZED.
                     # If you see the wrong camera, try 0, 1 or 2.
USE_ZED = True       # True = ZED (keeps only the left lens). False = normal webcam.
AUTO_INTERVAL = 0.5  # seconds between auto-saved photos
# -----------------------------------------------------------------------------

SAVE_DIR = os.path.join("dataset", "images")
os.makedirs(SAVE_DIR, exist_ok=True)

# Continue numbering from photos already in the folder.
count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Could not open camera {CAMERA_INDEX}. Try another CAMERA_INDEX (0, 1, 2).")
    raise SystemExit(1)

if USE_ZED:
    # The ZED sends both lenses side by side in one wide image (2 x 1280 x 720).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ok, test = cap.read()
if not ok:
    print("Camera opened but returned no image.")
    raise SystemExit(1)
print(f"Camera {CAMERA_INDEX} gives images of {test.shape[1]} x {test.shape[0]} pixels.")
if USE_ZED and test.shape[1] < 2 * test.shape[0]:
    print("Warning: this does not look like a side-by-side ZED image. "
          "Check CAMERA_INDEX, or set USE_ZED = False.")

auto = False
last_save = 0.0


def save(img):
    global count, last_save
    path = os.path.join(SAVE_DIR, f"pad_{count:04d}.jpg")
    cv2.imwrite(path, img)
    count += 1
    last_save = time.time()


while True:
    ok, frame = cap.read()
    if not ok:
        break

    if USE_ZED:
        frame = frame[:, : frame.shape[1] // 2]  # keep only the left lens

    if auto and time.time() - last_save >= AUTO_INTERVAL:
        save(frame)

    # Draw the overlay on a copy so saved photos stay clean.
    view = frame.copy()
    status = "AUTO ON" if auto else "auto off"
    cv2.putText(view, f"Saved: {count}  [{status}]  SPACE=save a=auto q=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Capture training photos", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        save(frame)
    elif key == ord("a"):
        auto = not auto
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done. {count} photos in {SAVE_DIR}")
