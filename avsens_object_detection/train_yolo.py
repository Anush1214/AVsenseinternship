from ultralytics import YOLO

# -------------------------------
# Train on COCO dataset
# -------------------------------
#coco_model = YOLO("yolov8n.pt")

#coco_model.train(
#    data="yolo_coco/data.yaml",
#    epochs=30,
#    imgsz=640,
#    project="runs/detect",
#    name="coco_train"
#)

# -------------------------------
# Train on Open Images dataset
# -------------------------------
oi_model = YOLO("yolov8n.pt")  # NEW instance

oi_model.train(
    data="yolo_openimages/data.yaml",
    epochs=30,
    imgsz=640,
    project="runs/detect",
    name="openimages_train"
)