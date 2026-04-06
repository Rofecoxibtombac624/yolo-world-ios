"""
YOLO-World → truly open-vocabulary CoreML export for iOS

Produces:
    YOLOWorldDetector.mlpackage   detector: image + text embedding → raw detections
    YOLOWorldText.mlpackage       CLIP text encoder: tokens → embedding
    clip_tokenizer.json           BPE vocab for on-device tokenization

The detector accepts a text embedding at runtime — no baked-in vocabulary.
User can type any word and get detections immediately.

Usage:
    pip install ultralytics>=8.3 coremltools>=7.2 torch clip
    python export_for_ios.py
"""

import json
import shutil
import pathlib
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2fAttn
from ultralytics.nn.modules.head import WorldDetect

IMG_SIZE = 640

print("Loading YOLO-World...")
yolo = YOLO("yolov8s-worldv2.pt")
wm = yolo.model
wm.eval()

# Set to single-class mode for export (nc=1: one score per detection)
wm.set_classes(["object"])
wm.model[-1].export = True

from ultralytics.nn.text_model import build_text_model
clip_wrapper = build_text_model("clip:ViT-B/32", device="cpu")
clip_inner = clip_wrapper.model
clip_inner.eval()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Detector — accepts text embedding as a dynamic input
# ─────────────────────────────────────────────────────────────────────────────

class DynamicYOLOWorld(nn.Module):
    """Full YOLO-World detector that accepts text embeddings at runtime."""
    def __init__(self, world_model):
        super().__init__()
        self.layers = world_model.model
        self.save = world_model.save

    def forward(self, x: torch.Tensor, txt_feats: torch.Tensor):
        # x:         [1, 3, 640, 640]  image
        # txt_feats: [1, 1, 512]       L2-normalised CLIP text embedding
        txt = txt_feats.to(dtype=x.dtype).expand(x.shape[0], -1, -1)
        ori_txt = txt.clone()
        y = []
        for m in self.layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, C2fAttn):
                x = m(x, txt)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt)
            else:
                x = m(x)
            y.append(x if m.i in self.save else None)
        return x


print("\n── Detector ──────────────────────────────────────────────────────────")
det_wrapper = DynamicYOLOWorld(wm).eval()

dummy_img = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
dummy_txt = torch.rand(1, 1, 512)

with torch.no_grad():
    out = det_wrapper(dummy_img, dummy_txt)
    print(f"Output shape: {out.shape}")  # [1, 5, 8400] — 4 box + 1 score

print("Tracing...")
with torch.no_grad():
    traced_det = torch.jit.trace(det_wrapper, (dummy_img, dummy_txt))

print("Converting to CoreML...")
det_mlmodel = ct.convert(
    traced_det,
    inputs=[
        ct.ImageType(
            name="image",
            shape=(1, 3, IMG_SIZE, IMG_SIZE),
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout=ct.colorlayout.RGB,
        ),
        ct.TensorType(name="txt_feats", shape=(1, 1, 512)),
    ],
    outputs=[
        ct.TensorType(name="detections"),
    ],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS16,
)

det_mlmodel.short_description = "YOLO-World detector — image + text embedding → detections"
det_mlmodel.input_description["image"] = "RGB camera frame 640x640"
det_mlmodel.input_description["txt_feats"] = "CLIP text embedding [1, 1, 512] from YOLOWorldText"
det_mlmodel.output_description["detections"] = "Raw detections [1, 5, 8400]: 4 box (cxcywh) + 1 score"

out_path = pathlib.Path("YOLOWorldDetector.mlpackage")
if out_path.exists():
    shutil.rmtree(out_path)
det_mlmodel.save(str(out_path))
print("✓ YOLOWorldDetector.mlpackage")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text encoder
# ─────────────────────────────────────────────────────────────────────────────

class TextEncoderWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens.long()).type(self.ln_final.weight.dtype)
        x = x + self.positional_embedding.type(self.ln_final.weight.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.ln_final.weight.dtype)
        eot_indices = tokens.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection
        x = x / x.norm(dim=-1, keepdim=True)
        return x.float()


print("\n── Text encoder ──────────────────────────────────────────────────────")
text_encoder = TextEncoderWrapper(clip_inner).eval()

import clip as clip_lib
dummy_tokens = clip_lib.tokenize(["person"]).int()

with torch.no_grad():
    emb = text_encoder(dummy_tokens)
    print(f"Output shape: {emb.shape}")  # [1, 512]

print("Tracing...")
with torch.no_grad():
    traced_text = torch.jit.trace(text_encoder, dummy_tokens)

print("Converting to CoreML...")
text_mlmodel = ct.convert(
    traced_text,
    inputs=[ct.TensorType(name="tokens", shape=(1, 77), dtype=np.int32)],
    outputs=[ct.TensorType(name="embedding")],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS16,
)
text_mlmodel.short_description = "CLIP text encoder — any word → embedding"
text_mlmodel.input_description["tokens"] = "CLIP token ids padded to length 77"
text_mlmodel.output_description["embedding"] = "L2-normalised text embedding [1, 512]"

out_path = pathlib.Path("YOLOWorldText.mlpackage")
if out_path.exists():
    shutil.rmtree(out_path)
text_mlmodel.save(str(out_path))
print("✓ YOLOWorldText.mlpackage")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tokenizer vocab
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Tokenizer vocab ───────────────────────────────────────────────────")
from clip.simple_tokenizer import SimpleTokenizer
tokenizer = SimpleTokenizer()
tokenizer_data = {
    "vocab": {k: v for k, v in tokenizer.encoder.items()},
    "merges": [" ".join(m) for m in tokenizer.bpe_ranks.keys()],
    "sot": 49406,
    "eot": 49407,
    "length": 77,
}
with open("clip_tokenizer.json", "w") as f:
    json.dump(tokenizer_data, f)
print("✓ clip_tokenizer.json")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Done. Add these files to your Xcode project:

  YOLOWorldDetector.mlpackage   detector (image + embedding → boxes)
  YOLOWorldText.mlpackage       text encoder (tokens → embedding)
  clip_tokenizer.json           tokenizer vocab

Runtime flow on device:
  user types ANY word
    → tokenize (pure Swift, ~0ms)
    → YOLOWorldText encodes it (~15ms, cached)
    → YOLOWorldDetector every frame with that embedding (~30ms)
    → NMS in Swift
    → draw boxes

No vocabulary list. No re-export. Fully open-vocabulary.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
