from ultralytics import YOLO

# ==============================
# ABSOLUTE PATHS (FINAL FIX)
# ==============================
COCO_WEIGHTS = r"D:\AVsenseinternship\avsens_object_detection\runs\detect\runs\detect\coco_train\weights\best.pt"
COCO_DATA = r"D:\AVsenseinternship\avsens_object_detection\yolo_coco\data.yaml"

print("\n=== Evaluating COCO-trained model on COCO validation ===")

coco_model = YOLO(COCO_WEIGHTS)

metrics = coco_model.val(
    data=COCO_DATA,
    split="val",
    imgsz=640
)

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP@50:", metrics.box.map50)
print("mAP@50-95:", metrics.box.map)


# ==================================================
# ABSOLUTE PATHS – OPEN IMAGES EVALUATION
# ==================================================
OPENIMAGES_WEIGHTS = r"D:\AVsenseinternship\avsens_object_detection\runs\detect\runs\detect\openimages_train\weights\best.pt"
OPENIMAGES_DATA = r"D:\AVsenseinternship\avsens_object_detection\yolo_openimages\data.yaml"

print("\n=== Evaluating OpenImages-trained model on OpenImages validation ===")

oi_model = YOLO(OPENIMAGES_WEIGHTS)

oi_metrics = oi_model.val(
    data=OPENIMAGES_DATA,
    split="val",
    imgsz=640
)

print("OpenImages Precision:", oi_metrics.box.mp)
print("OpenImages Recall:", oi_metrics.box.mr)
print("OpenImages mAP@50:", oi_metrics.box.map50)
print("OpenImages mAP@50-95:", oi_metrics.box.map)