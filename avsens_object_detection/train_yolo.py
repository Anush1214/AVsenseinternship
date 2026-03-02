from ultralytics import YOLO


coco_model = YOLO("yolov8n.pt")

coco_model.train(
    data="yolo_coco/data.yaml",
    epochs=30,
    imgsz=640,
    project="runs/detect",
    name="coco_train"
)


oi_model = YOLO("yolov8n.pt")

oi_model.train(
    data="yolo_openimages/data.yaml",
    epochs=30,
    imgsz=640,
    project="runs/detect",
    name="openimages_train"
)


custom_model = YOLO("yolov8n.pt")

custom_model.train(
    data="Internship_task-2-1/data.yaml",
    epochs=30,
    imgsz=640,
    project="runs/detect",
    name="custom_pedestrian_train"
)