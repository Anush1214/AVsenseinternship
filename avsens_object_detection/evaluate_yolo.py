from ultralytics import YOLO

# ==================================================
# COCO EVALUATION
# ==================================================
COCO_WEIGHTS = r"D:\AVsenseinternship\avsens_object_detection\runs\detect\runs\detect\coco_train\weights\best.pt"
COCO_DATA = r"D:\AVsenseinternship\avsens_object_detection\yolo_coco\data.yaml"

print("\n=== Evaluating COCO-trained model on COCO validation ===")

coco_model = YOLO(COCO_WEIGHTS)
coco_metrics = coco_model.val(
    data=COCO_DATA,
    split="val",
    imgsz=640
)

print("COCO Precision:", coco_metrics.box.mp)
print("COCO Recall:", coco_metrics.box.mr)
print("COCO mAP@50:", coco_metrics.box.map50)
print("COCO mAP@50-95:", coco_metrics.box.map)


# ==================================================
# OPEN IMAGES EVALUATION
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


# ==================================================
# CUSTOM PEDESTRIAN DATASET EVALUATION (NEW)
# ==================================================
CUSTOM_WEIGHTS = r"D:\AVsenseinternship\avsens_object_detection\runs\detect\runs\detect\custom_pedestrian_train\weights\best.pt"
CUSTOM_DATA = r"D:\AVsenseinternship\avsens_object_detection\Internship_task-2-1\data.yaml"

print("\n=== Evaluating CUSTOM PEDESTRIAN model on custom validation ===")

custom_model = YOLO(CUSTOM_WEIGHTS)
custom_metrics = custom_model.val(
    data=CUSTOM_DATA,
    split="val",
    imgsz=640
)

print("Custom Precision:", custom_metrics.box.mp)
print("Custom Recall:", custom_metrics.box.mr)
print("Custom mAP@50:", custom_metrics.box.map50)
print("Custom mAP@50-95:", custom_metrics.box.map)