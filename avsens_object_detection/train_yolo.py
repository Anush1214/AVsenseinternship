from ultralytics import YOLO

# ===============================
# 1. Train on COCO dataset
# ===============================
#coco_model = YOLO("yolov8n.pt")

#coco_model.train(
#    data="yolo_coco/data.yaml",
#    epochs=30,
#    imgsz=640,
#    project="runs/detect",
#    name="coco_train"
#)

# ===============================
# 2. Train on Open Images dataset
# ===============================
#oi_model = YOLO("yolov8n.pt")

#oi_model.train(
#    data="yolo_openimages/data.yaml",
#    epochs=30,
#    imgsz=640,
#    project="runs/detect",
#    name="openimages_train"
#)

# ===============================
# 3. Train on CUSTOM PEDESTRIAN dataset
# (Internship_task-2-1)
# ===============================
custom_model = YOLO("yolov8n.pt")

custom_model.train(
    data="Internship_task-2-1/data.yaml",
    epochs=30,
    imgsz=640,
    project="runs/detect",
    name="custom_pedestrian_train"
)