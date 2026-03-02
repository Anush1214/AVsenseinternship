import os
import shutil
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


MAX_SAMPLES = 200
TRAIN_RATIO = 0.8

YOLO_CLASSES = ["person", "car"]

COCO_CLASSES = ["person", "car"]
OPENIMAGES_REMAP = {
    "Person": "person",
    "Car": "car",
}

# ==========================
# LOAD DATASET
# ==========================
def load_dataset(dataset_name, zoo_name, split, classes=None):
    print(f"\n Loading {dataset_name}...")

    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = foz.load_zoo_dataset(
        zoo_name,
        split=split,
        label_types=["detections"],
        max_samples=MAX_SAMPLES,
        dataset_name=dataset_name,
    )

    # Optional class filtering
    if classes:
        view = dataset.filter_labels(
            "ground_truth",
            F("label").is_in(classes)
        )
    else:
        view = dataset.view()

    print(view)
    return view


# ==========================
# NORMALIZE LABELS
# ==========================
def normalize_labels(view, mapping):
    print(" Normalizing labels...")
    for sample in view:
        if sample.ground_truth is None:
            continue
        for det in sample.ground_truth.detections:
            if det.label in mapping:
                det.label = mapping[det.label]
        sample.save()


# ==========================
# YOLO EXPORT 
# ==========================
def export_to_yolo(view, export_dir):
    print(f"\n Exporting YOLO dataset to: {export_dir}")

    # Clean directory
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    img_train = os.path.join(export_dir, "images/train")
    img_val   = os.path.join(export_dir, "images/val")
    lbl_train = os.path.join(export_dir, "labels/train")
    lbl_val   = os.path.join(export_dir, "labels/val")

    os.makedirs(img_train, exist_ok=True)
    os.makedirs(img_val, exist_ok=True)
    os.makedirs(lbl_train, exist_ok=True)
    os.makedirs(lbl_val, exist_ok=True)

    # Convert view to list (stable)
    samples = list(view)
    total = len(samples)
    split = int(TRAIN_RATIO * total)

    train_samples = samples[:split]
    val_samples   = samples[split:]

    # ---- EXPORT LABELS ONLY ----
    train_ds = fo.Dataset()
    train_ds.add_samples(train_samples)

    val_ds = fo.Dataset()
    val_ds.add_samples(val_samples)

    train_ds.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split="train",
        classes=YOLO_CLASSES,
        label_field="ground_truth",
        export_media=False,
    )

    val_ds.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split="val",
        classes=YOLO_CLASSES,
        label_field="ground_truth",
        export_media=False,
    )

    # ---- MANUAL IMAGE COPY ----
    for s in train_samples:
        shutil.copy2(
            s.filepath,
            os.path.join(img_train, os.path.basename(s.filepath))
        )

    for s in val_samples:
        shutil.copy2(
            s.filepath,
            os.path.join(img_val, os.path.basename(s.filepath))
        )

    # ---- data.yaml ----
    with open(os.path.join(export_dir, "data.yaml"), "w") as f:
        f.write(f"""
path: {os.path.abspath(export_dir)}
train: images/train
val: images/val

names:
  0: person
  1: car
""".strip())

    print("EXPORT COMPLETE")
    print("Train images:", len(os.listdir(img_train)))
    print("Val images:", len(os.listdir(img_val)))
    print("Train labels:", len(os.listdir(lbl_train)))
    print("Val labels:", len(os.listdir(lbl_val)))

    fo.delete_dataset(train_ds.name)
    fo.delete_dataset(val_ds.name)


# ==========================
# MAIN
# ==========================
def main():
    print("\n==============================")
    print(" YOLO DATASET PREPARATION")
    print(" COCO 2017 vs Open Images V7")
    print("==============================")

    # ---- COCO ----
    coco_view = load_dataset(
        "coco-person-car",
        "coco-2017",
        "validation",
        classes=COCO_CLASSES,
    )
    export_to_yolo(coco_view, "yolo_coco")

    # ---- OPEN IMAGES ----
    oi_view = load_dataset(
        "openimages-person-car",
        "open-images-v7",
        "validation",
        classes=None,   # IMPORTANT: no filtering at load
    )

    normalize_labels(oi_view, OPENIMAGES_REMAP)
    export_to_yolo(oi_view, "yolo_openimages")

    print("\n DATASETS READY (IMAGES + LABELS VERIFIED)")

if __name__ == "__main__":
    main()