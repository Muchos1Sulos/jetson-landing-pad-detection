"""
Creates an AprilTag image (family 36h11, ID 0) you can print or open on your phone.

Run:  python make_tag.py
Output: tag36h11_id0.png
"""

import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
tag = cv2.aruco.generateImageMarker(dictionary, 0, 400)  # ID 0, 400x400 pixels

# Add a white border so the detector can see the tag's black edge.
tag = cv2.copyMakeBorder(tag, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
cv2.imwrite("tag36h11_id0.png", tag)
print("Saved tag36h11_id0.png")
