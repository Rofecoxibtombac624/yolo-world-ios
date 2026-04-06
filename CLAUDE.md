# YOLO-World iOS

## Build

```bash
xcodegen generate
open YOLOWorldApp.xcodeproj
# Build & run on physical device (camera + LiDAR required)
```

## Re-export CoreML models

```bash
pip install ultralytics>=8.3 coremltools>=7.2 torch clip
python export_for_ios.py
```

Produces `YOLOWorldDetector.mlpackage`, `YOLOWorldText.mlpackage`, and `clip_tokenizer.json`.

## Architecture

- `ObjectDetector` -- camera capture, CoreML inference, LiDAR depth
- `IntentParser` -- on-device Qwen3-0.6B via LLM.swift, parses natural language into `DetectionIntent`
- `CLIPTokenizer` -- BPE tokenizer for CLIP text encoder
- Two CoreML models: detector (runs every frame) + text encoder (runs on user input)
- Detector accepts dynamic text embeddings -- no fixed vocabulary

## Key decisions

- Text encoder is split from detector so vocabulary isn't baked in at export time
- Fresh LLM instance per parse call to avoid KV cache corruption
- `/no_think` prefix for Qwen3 to skip chain-of-thought and go straight to JSON
- Confidence threshold: 0.15, NMS IoU: 0.45
- Camera prefers LiDAR device, falls back to dual-wide, then wide-angle
- Pixel buffers converted to 32BGRA before CoreML inference
