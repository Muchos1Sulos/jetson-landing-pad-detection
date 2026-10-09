"""
Step 6 of Project 1: run your TensorRT engine LIVE on the Jetson with the ZED.

Everything runs on the Jetson: camera -> preprocess -> TensorRT (GPU) -> boxes.
It times every stage separately, so you can see where the time goes.

Two ways to run it (in ~/sultan on the Jetson; stop stream_capture.py first,
because only one program can use the camera at a time):

  1) Live view in your Mac's browser:
         python3 trt_live.py
     then open  http://192.168.55.1:8000   (Ctrl+C to stop)

  2) Benchmark only (no video), prints a timing table after N frames:
         python3 trt_live.py --bench 300

Options:
  --engine best_fp32.engine   use another engine (default: best_fp16.engine)
  --conf 0.5                  minimum confidence to show a detection

No pycuda / cuda-python needed: GPU memory is handled by calling the CUDA
runtime library (libcudart) directly through Python's ctypes.
"""

import argparse
import ctypes
import glob
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

INPUT_SIZE = 640


# ----------------------------------------------------------------------------
# Tiny CUDA runtime wrapper (ctypes)
# ----------------------------------------------------------------------------
class CudaRT:
    H2D, D2H = 1, 2  # cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost

    def __init__(self):
        candidates = ["libcudart.so"] + sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"))
        self.lib = None
        for path in candidates:
            try:
                self.lib = ctypes.CDLL(path)
                break
            except OSError:
                continue
        if self.lib is None:
            raise SystemExit("Could not load libcudart. Is CUDA installed (/usr/local/cuda)?")
        L = self.lib
        L.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        L.cudaFree.argtypes = [ctypes.c_void_p]
        L.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        L.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        L.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        L.cudaGetErrorString.argtypes = [ctypes.c_int]
        L.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, code, what):
        if code != 0:
            msg = self.lib.cudaGetErrorString(code).decode()
            raise RuntimeError(f"CUDA error in {what}: {msg} ({code})")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr

    def free(self, ptr):
        self.lib.cudaFree(ptr)

    def to_gpu(self, dev_ptr, host_array):
        self._check(self.lib.cudaMemcpy(dev_ptr, host_array.ctypes.data, host_array.nbytes,
                                        self.H2D), "cudaMemcpy H2D")

    def to_cpu(self, host_array, dev_ptr):
        self._check(self.lib.cudaMemcpy(host_array.ctypes.data, dev_ptr, host_array.nbytes,
                                        self.D2H), "cudaMemcpy D2H")

    def stream(self):
        s = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s

    def sync(self, stream):
        self._check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


# ----------------------------------------------------------------------------
# TensorRT engine
# ----------------------------------------------------------------------------
class TRTModel:
    def __init__(self, engine_path):
        import tensorrt as trt  # imported here so the rest of the file can be tested elsewhere

        self.cuda = CudaRT()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"Could not load {engine_path}.")
        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.stream()

        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.zeros(shape, dtype=dtype)
            dev = self.cuda.malloc(host.nbytes)
            self.context.set_tensor_address(name, dev.value)
            entry = {"name": name, "host": host, "dev": dev}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)
        for e in self.inputs + self.outputs:
            print(f"  tensor {e['name']}: shape {e['host'].shape}, {e['host'].dtype}")

    def infer(self, blob):
        """blob: float32 array (1, 3, 640, 640). Returns the first output array."""
        inp = self.inputs[0]
        np.copyto(inp["host"], blob)
        self.cuda.to_gpu(inp["dev"], inp["host"])
        self.context.execute_async_v3(self.stream.value)
        self.cuda.sync(self.stream)
        out = self.outputs[0]
        self.cuda.to_cpu(out["host"], out["dev"])
        return out["host"]


# ----------------------------------------------------------------------------
# Pre- and post-processing (CPU, NumPy/OpenCV)
# ----------------------------------------------------------------------------
def preprocess(frame):
    """BGR image -> (1, 3, 640, 640) float32 RGB in [0, 1].
    Uses a plain stretch-resize, matching Roboflow's 'Stretch to 640x640'."""
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(blob.transpose(2, 0, 1)[None])


def postprocess(output, frame_w, frame_h, conf_thres=0.5, iou_thres=0.45):
    """YOLO11 output (1, 4+classes, 8400) -> list of (x1, y1, x2, y2, conf) in frame pixels."""
    preds = output[0]
    if preds.shape[0] < preds.shape[1]:      # (5, 8400) -> (8400, 5)
        preds = preds.T
    scores = preds[:, 4:].max(axis=1)
    keep = scores >= conf_thres
    preds, scores = preds[keep], scores[keep]
    if len(preds) == 0:
        return []

    sx, sy = frame_w / INPUT_SIZE, frame_h / INPUT_SIZE
    cx, cy, w, h = preds[:, 0] * sx, preds[:, 1] * sy, preds[:, 2] * sx, preds[:, 3] * sy
    x1, y1 = cx - w / 2, cy - h / 2
    boxes_xywh = np.stack([x1, y1, w, h], axis=1)

    idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf_thres, iou_thres)
    idx = np.array(idx).flatten() if len(idx) else []
    return [(int(x1[i]), int(y1[i]), int(x1[i] + w[i]), int(y1[i] + h[i]), float(scores[i]))
            for i in idx]


def draw(frame, dets, stats):
    for x1, y1, x2, y2, conf in dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"pad {conf:.0%}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    lines = [f"FPS: {stats['fps']:.1f}",
             f"TensorRT: {stats['infer']:.1f} ms   pre: {stats['pre']:.1f} ms   "
             f"post: {stats['post']:.1f} ms"]
    for i, txt in enumerate(lines):
        y = 30 + 30 * i
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return frame


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
def open_zed(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {source}. Is stream_capture.py still running?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def left_lens(frame):
    if frame.shape[1] >= 2 * frame.shape[0]:
        return frame[:, : frame.shape[1] // 2]
    return frame


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
STAGES = ("capture", "pre", "infer", "post")


def run(args, shared=None):
    print(f"Loading {args.engine} ...")
    model = TRTModel(args.engine)
    cap = open_zed(args.camera)

    # Warm up the GPU so the first measurements are not slow outliers.
    dummy = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), np.float32)
    for _ in range(10):
        model.infer(dummy)

    sums = {k: 0.0 for k in STAGES}
    n = 0
    fps = 0.0
    t_prev = time.perf_counter()
    stats = {"fps": 0.0, "pre": 0.0, "infer": 0.0, "post": 0.0}
    t_start = time.perf_counter()

    try:
        while shared is None or shared["running"]:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            frame = left_lens(frame)
            t1 = time.perf_counter()
            blob = preprocess(frame)
            t2 = time.perf_counter()
            output = model.infer(blob)
            t3 = time.perf_counter()
            dets = postprocess(output, frame.shape[1], frame.shape[0], args.conf)
            t4 = time.perf_counter()

            times = {"capture": t1 - t0, "pre": t2 - t1, "infer": t3 - t2, "post": t4 - t3}
            for k in STAGES:
                sums[k] += times[k]
            n += 1

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            stats = {"fps": fps, "pre": times["pre"] * 1000, "infer": times["infer"] * 1000,
                     "post": times["post"] * 1000}

            if shared is not None:
                view = draw(frame.copy(), dets, stats)
                ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
                with shared["lock"]:
                    if ok:
                        shared["jpeg"] = jpg.tobytes()
                    shared["stats"] = {**stats, "detections": len(dets),
                                       "best": max((d[4] for d in dets), default=0.0)}

            if args.bench and n >= args.bench:
                break
    except KeyboardInterrupt:
        pass  # Ctrl+C: stop the loop but still print the timing report
    finally:
        cap.release()

    total = time.perf_counter() - t_start
    print_report(args.engine, sums, n, total)


def print_report(engine, sums, n, total):
    if n == 0:
        return
    avg = {k: sums[k] / n * 1000 for k in STAGES}
    work = avg["pre"] + avg["infer"] + avg["post"]
    print(f"\n===== Pipeline timing: {engine}, {n} frames =====")
    print(f"{'Stage':<22}{'Avg ms':>10}{'Share of work':>16}")
    names = {"pre": "Preprocess (CPU)", "infer": "TensorRT (GPU+copies)",
             "post": "Postprocess (CPU)"}
    for k in ("pre", "infer", "post"):
        print(f"{names[k]:<22}{avg[k]:>10.2f}{avg[k] / work:>15.0%}")
    print(f"{'Processing total':<22}{work:>10.2f}   -> max {1000 / work:.0f} FPS if camera were instant")
    print(f"{'Camera wait/capture':<22}{avg['capture']:>10.2f}")
    print(f"End-to-end: {n / total:.1f} FPS (limited by the slowest of camera and processing)")


# ----------------------------------------------------------------------------
# Web view
# ----------------------------------------------------------------------------
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TensorRT live</title>
<style>body{font-family:-apple-system,sans-serif;background:#111;color:#eee;margin:0;padding:16px;text-align:center}
img{max-width:100%;border-radius:8px}#s{font-size:18px;margin:10px}</style></head><body>
<img src="/stream?t=1" alt="Live TensorRT detections" width="1280" height="720" style="height:auto;background:#000"><div id="s"></div>
<script>setInterval(()=>fetch('/stats').then(r=>r.json()).then(s=>{document.getElementById('s').textContent=
`FPS ${s.fps.toFixed(1)} | TensorRT ${s.infer.toFixed(1)} ms | pre ${s.pre.toFixed(1)} ms | post ${s.post.toFixed(1)} ms | pads: ${s.detections}`;}),500);
</script></body></html>"""


def serve(shared, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

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
            elif self.path == "/stats":
                with shared["lock"]:
                    body = json.dumps(shared["stats"]).encode()
                self._send(body, "application/json")
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    while shared["running"]:
                        with shared["lock"]:
                            jpg = shared["jpeg"]
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(1 / 20)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    p = argparse.ArgumentParser(description="Live TensorRT landing-pad detection on Jetson")
    p.add_argument("--engine", default="best_fp16.engine")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--bench", type=int, default=0, help="benchmark N frames, no web view")
    p.add_argument("--camera", default=0, type=lambda s: int(s) if s.isdigit() else s)
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    if args.bench:
        run(args)
        return

    shared = {"lock": threading.Lock(), "jpeg": None, "running": True,
              "stats": {"fps": 0, "infer": 0, "pre": 0, "post": 0, "detections": 0, "best": 0}}
    server = serve(shared, args.port)
    print(f"Open http://192.168.55.1:{args.port} in your Mac's browser. Ctrl+C to stop.")
    try:
        run(args, shared)
    except KeyboardInterrupt:
        pass
    finally:
        shared["running"] = False
        server.shutdown()


if __name__ == "__main__":
    main()
