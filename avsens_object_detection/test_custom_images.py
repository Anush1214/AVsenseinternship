from ultralytics import YOLO
import os

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "runs/detect/train/weights/best.pt"
IMAGE_SOURCE = "custom_test_images"
CONFIDENCE = 0.25
IMG_SIZE = 640
OUTPUT_DIR = "custom_predictions"

# ==========================
# LOAD MODEL
# ==========================
model = YOLO(MODEL_PATH)

# ==========================
# RUN PREDICTION
# ==========================
results = model.predict(
    source=IMAGE_SOURCE,
    conf=CONFIDENCE,
    imgsz=IMG_SIZE,
    save=True,
    project=OUTPUT_DIR,
    name="predictions"
)

print("✅ Inference complete")
print(f"📂 Results saved in: {os.path.abspath(OUTPUT_DIR)}")