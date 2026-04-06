"""
YOLO-World → CoreML export for iOS demo
Produces: YOLOWorld.mlpackage

Run once on any Mac or Linux machine. Takes ~5 minutes.
The output file is everything the iOS app needs.

Requirements:
    pip install ultralytics>=8.3 coremltools>=7.2
"""

from ultralytics import YOLO

# ── Vocabulary ────────────────────────────────────────────────────────────────
# These are the only words the app will detect. Add or remove freely,
# then re-run this script and replace YOLOWorld.mlpackage in Xcode.

VOCAB = [
    # People
    "person", "crowd", "student", "speaker", "presenter",
    # Furniture
    "chair", "table", "desk", "podium", "bench",
    # Tech
    "laptop", "phone", "bag", "backpack", "camera",
    "projector", "screen", "microphone", "tablet",
    # Conference
    "badge", "poster", "banner",
    # Objects
    "bottle", "cup", "notebook", "pen", "book",
    # Wildcards
    "hand", "hat", "glasses", "jacket",
    # Custom
    "cube", "rubiks cube",
    "earbuds", "earbuds case", "wireless earphones", "earphones",
]

# ── Model size options ────────────────────────────────────────────────────────
# "yolov8s-worldv2.pt"  →  small,  ~30 FPS on A17 Pro, good accuracy
# "yolov8m-worldv2.pt"  →  medium, ~15 FPS on A17 Pro, better accuracy
# Start with small. Switch to medium only if detections look poor in the hall.

MODEL_NAME = "yolov8s-worldv2.pt"
IMG_SIZE   = 640      # do not change — CoreML export is fixed-shape
OUTPUT     = "YOLOWorld.mlpackage"

# ── Export ────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
model = YOLO(MODEL_NAME)

print(f"Setting vocabulary ({len(VOCAB)} classes)...")
model.set_classes(VOCAB)

print("Exporting to CoreML (this takes a few minutes)...")
model.export(
    format="coreml",
    imgsz=IMG_SIZE,
    simplify=True,      # cleans up the ONNX graph before CoreML conversion
    nms=True,           # bakes NMS into the model — one less thing for Swift
)

# Ultralytics writes the file next to the weights, rename it
import shutil, pathlib

exported = pathlib.Path(MODEL_NAME).with_suffix("").name + ".mlpackage"
if pathlib.Path(OUTPUT).exists():
    shutil.rmtree(OUTPUT)
shutil.move(exported, OUTPUT)

print(f"\n✓ Done → {OUTPUT}")
print(f"  Vocabulary : {len(VOCAB)} classes")
print(f"  Input      : RGB image, 640×640")
print(f"  Outputs    : boxes [N,4]  cxcywh normalised 0–1")
print(f"               scores [N]   confidence 0–1")
print(f"               labels [N]   int index into VOCAB")
print(f"\nAdd {OUTPUT} to your Xcode project and you're good to go.")