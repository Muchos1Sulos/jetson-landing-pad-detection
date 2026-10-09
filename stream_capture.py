"""
Live ZED view in your Mac's browser + photo capture. Runs on the Jetson.

1. On the Jetson:   python3 stream_capture.py
2. On your Mac, open a browser at:   http://192.168.55.1:8000
   (use the same address you used for ssh)
3. Click "Save photo", or "Auto-save" to save a photo every 0.5 s.
   Click "Detect AprilTag" to see tag detections on the live view
   (saved photos never contain the drawings).
4. Stop with Ctrl+C in the Jetson terminal.

Photos are saved on the Jetson in: dataset/images/
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

# ---- Settings ----------------------------------------------------------------
CAMERA = sys.argv[1] if len(sys.argv) > 1 else 0   # 0 = /dev/video0 (the ZED)
USE_ZED = True        # keep only the left lens of the side-by-side ZED image
AUTO_INTERVAL = 0.5   # seconds between auto-saved photos
PORT = 8000
# ------------------------------------------------------------------------------

SAVE_DIR = os.path.join("dataset", "images")
os.makedirs(SAVE_DIR, exist_ok=True)

state = {
    "frame": None,       # latest camera image (left lens only)
    "jpeg": None,        # latest image as JPEG bytes, for the browser
    "count": len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")]),
    "auto": False,
    "detect": False,     # draw AprilTag detections on the live view
    "tags": "",          # text describing the last detection, shown on the page
    "last_save": 0.0,
    "running": True,
}
lock = threading.Lock()


def make_detector():
    """AprilTag 36h11 detector that works on both newer and older OpenCV versions."""
    if not hasattr(cv2, "aruco"):
        print("This OpenCV has no aruco module, so AprilTag detection is unavailable.")
        return None
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "ArucoDetector"):            # OpenCV 4.7 and newer
        det = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        return lambda gray: det.detectMarkers(gray)
    params = cv2.aruco.DetectorParameters_create()     # older OpenCV
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


detect_tags = make_detector()


def save_photo():
    with lock:
        frame = state["frame"]
        if frame is None:
            return
        path = os.path.join(SAVE_DIR, f"pad_{state['count']:04d}.jpg")
        cv2.imwrite(path, frame)
        state["count"] += 1
        state["last_save"] = time.time()


def camera_loop():
    cam = int(CAMERA) if str(CAMERA).isdigit() else CAMERA
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA}.")
        state["running"] = False
        return
    if USE_ZED:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    printed = False
    while state["running"]:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        if not printed:
            print(f"Camera gives {frame.shape[1]} x {frame.shape[0]} images.")
            printed = True
        if USE_ZED and frame.shape[1] >= 2 * frame.shape[0]:
            frame = frame[:, : frame.shape[1] // 2]

        # The browser view gets drawings; the saved photo (frame) stays clean.
        view = frame
        tags_text = ""
        if state["detect"] and detect_tags is not None:
            view = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t0 = time.perf_counter()
            corners, ids, _ = detect_tags(gray)
            ms = (time.perf_counter() - t0) * 1000
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(view, corners, ids)
                found = []
                for c, tag_id in zip(corners, ids.flatten()):
                    cx, cy = c[0].mean(axis=0).astype(int)
                    cv2.circle(view, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                    found.append(f"ID {tag_id} at ({cx}, {cy})")
                tags_text = f"{', '.join(found)} | {ms:.1f} ms"
            else:
                tags_text = f"No tag | {ms:.1f} ms"
            cv2.putText(view, tags_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with lock:
            state["tags"] = tags_text
            state["frame"] = frame
            if ok:
                state["jpeg"] = jpg.tobytes()
            due = state["auto"] and time.time() - state["last_save"] >= AUTO_INTERVAL
        if due:
            save_photo()
    cap.release()


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZED capture</title>
<style>
 body{font-family:-apple-system,sans-serif;background:#111;color:#eee;margin:0;padding:16px;text-align:center}
 img{max-width:100%;border-radius:8px;background:#000}
 button{font-size:18px;padding:10px 20px;margin:8px;border-radius:8px;border:0;cursor:pointer}
 #save{background:#2d7ff9;color:#fff} #auto,#det{background:#444;color:#fff}
 #auto.on{background:#d33} #det.on{background:#1a9e55}
 #count{font-size:22px;margin:8px}
</style></head><body>
<img src="/stream" alt="Live ZED view">
<div id="count">Saved: 0</div>
<button id="save" onclick="fetch('/save',{method:'POST'}).then(upd)">Save photo (space)</button>
<button id="auto" onclick="fetch('/auto',{method:'POST'}).then(upd)">Auto-save: off (a)</button>
<button id="det" onclick="fetch('/detect',{method:'POST'}).then(upd)">Detect AprilTag: off (d)</button>
<div id="tags" style="font-size:18px;min-height:26px"></div>
<script>
function upd(){fetch('/status').then(r=>r.json()).then(s=>{
  document.getElementById('count').textContent='Saved: '+s.count;
  const b=document.getElementById('auto');
  b.textContent='Auto-save: '+(s.auto?'ON':'off')+' (a)'; b.classList.toggle('on',s.auto);
  const d=document.getElementById('det');
  d.textContent='Detect AprilTag: '+(s.detect?'ON':'off')+' (d)'; d.classList.toggle('on',s.detect);
  document.getElementById('tags').textContent=s.detect?s.tags:'';});}
document.addEventListener('keydown',e=>{
  if(e.code==='Space'){e.preventDefault();fetch('/save',{method:'POST'}).then(upd);}
  if(e.key==='a'){fetch('/auto',{method:'POST'}).then(upd);}
  if(e.key==='d'){fetch('/detect',{method:'POST'}).then(upd);}});
setInterval(upd,500); upd();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the terminal quiet

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/status":
            with lock:
                body = json.dumps({"count": state["count"], "auto": state["auto"],
                                   "detect": state["detect"], "tags": state["tags"]})
            self._send(body.encode(), "application/json")
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while state["running"]:
                    with lock:
                        jpg = state["jpeg"]
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode() +
                                         b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1 / 20)  # about 20 frames per second to the browser
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/save":
            save_photo()
        elif self.path == "/auto":
            with lock:
                state["auto"] = not state["auto"]
        elif self.path == "/detect":
            with lock:
                state["detect"] = not state["detect"]
        else:
            self.send_error(404)
            return
        self._send(b"ok", "text/plain")


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Open http://192.168.55.1:{PORT} in your Mac's browser. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    state["running"] = False
    server.server_close()
    print(f"Stopped. {state['count']} photos in {SAVE_DIR}")
