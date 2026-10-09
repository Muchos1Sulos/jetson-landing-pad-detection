# Real-Time Landing-Pad Detection on NVIDIA Jetson

A YOLO11 detector that finds a drone landing pad from a ZED stereo camera and runs in real time **on an NVIDIA Jetson Orin Nano with TensorRT**. Built for the autonomous delivery drone at ARC Purdue, where the existing AprilTag-based landing only worked up close.

![Live TensorRT detection on the Jetson](images/live_jetson.jpeg)

## Results

**Detection range: YOLO vs. AprilTag** (same ZED camera, same pad)

| Detector | Max detection distance |
|---|---|
| AprilTag (OpenCV, 36h11) | 2 m |
| **YOLO11n (this project)** | **5 m (2.5× farther)** |

![YOLO finds the pad where AprilTag cannot](images/yolo_vs_apriltag.jpeg)

**Model accuracy** (46 held-out validation images)

| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| **0.942** | 0.723 | 0.866 | 0.952 |

**TensorRT on Jetson Orin Nano** (JetPack 6.2, TensorRT 10.3, 15 W mode, `trtexec`)

| Engine | GPU compute (mean) | Latency (mean) | Throughput |
|---|---|---|---|
| FP32 | 11.28 ms | ~11.7 ms | 88 inferences/s |
| **FP16** | **6.39 ms** | **6.87 ms** | **156 inferences/s (1.77×)** |

**Live pipeline on the Jetson** (ZED → preprocess → TensorRT → postprocess, 300 frames)

| Stage | FP32 | FP16 |
|---|---|---|
| Preprocess (CPU) | 8.55 ms | 7.84 ms |
| TensorRT (GPU + copies) | 23.75 ms | **14.44 ms** |
| Postprocess (CPU) | 0.52 ms | 0.61 ms |
| Processing total | 32.82 ms (max 30 FPS) | **22.90 ms (max 44 FPS)** |
| End-to-end | 24.6 FPS | 24.6 FPS |

### What the measurements show
- **FP16 makes inference 1.6–1.8× faster** with no visible loss in detection quality.
- **End-to-end is camera-limited** (the ZED delivers ~25 FPS in this mode): both engines reach the same 24.6 FPS. FP16 still frees ~40% of each frame for other work on the drone.
- **CPU preprocessing is ~34% of processing time**, the next bottleneck. Moving it to custom CUDA kernels is the follow-up project.
- Inference in the live loop (14.4 ms) is slower than in `trtexec` (6.9 ms), because the GPU idles between camera frames and lowers its clock in 15 W mode, while `trtexec` keeps it busy.

## How it works

```
ZED camera ──► left lens 1280×720 ──► resize 640×640, RGB, /255 ──► TensorRT FP16 engine ──► NMS ──► landing-pad boxes
               (Jetson, OpenCV)        (CPU, NumPy)                  (GPU)                     (CPU)
```

1. **Data**: 232 photos captured with the ZED on the Jetson (`stream_capture.py`, viewed remotely in a browser), labeled in Roboflow, split 70/20/10.
2. **Training**: YOLO11n (Ultralytics) fine-tuned for 50 epochs on an Apple M4 Pro (`train_yolo.py`).
3. **Export**: PyTorch → ONNX (opset 18) → TensorRT engines built on the Jetson with `trtexec` (FP32 and FP16).
4. **Deployment**: `trt_live.py` runs the full pipeline on the Jetson. It loads the engine with the TensorRT Python API and manages GPU memory by calling the **CUDA runtime library directly through `ctypes`** (no PyCUDA needed), and it times every stage.

## Repository

| File | What it does | Runs on |
|---|---|---|
| `stream_capture.py` | Live ZED view in a browser, photo capture, optional AprilTag overlay | Jetson |
| `train_yolo.py` | Trains YOLO11n on the Roboflow export and prints metrics | Mac / PC |
| `detect_yolo.py` | Side-by-side YOLO vs. AprilTag live comparison | Mac / PC |
| `trt_live.py` | On-device TensorRT pipeline with web view and `--bench` timing | Jetson |
| `detect_apriltag.py`, `make_tag.py` | AprilTag baseline and tag generator | Any |

## Reproduce

```bash
# On a Mac/PC
pip install ultralytics
python train_yolo.py                       # expects the Roboflow export in pad_dataset/
yolo export model=runs/detect/pad_v1/weights/best.pt format=onnx imgsz=640

# On the Jetson (JetPack 6.x)
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best_fp16.engine --fp16
python3 trt_live.py                        # live view at http://<jetson-ip>:8000
python3 trt_live.py --bench 300            # timing table
```

## Limitations and next steps
- The dataset is small and mostly from one room; some pads are labeled as two boxes. Version 2 will add far-range, outdoor and top-down shots with one box per pad.
- **CUDA preprocessing kernels** (resize + color conversion + normalization fused into one kernel), profiled with Nsight Systems.
- INT8 quantization with calibration, and a threaded capture loop with a higher-FPS ZED mode.

## Hardware and software
NVIDIA Jetson Orin Nano (JetPack 6.2 / L4T R36.4.7, TensorRT 10.3, 15 W) · Stereolabs ZED (UVC, left lens) · Ultralytics YOLO11n · OpenCV · Roboflow
