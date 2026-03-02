import pandas as pd

# ==============================
# COCO RESULTS
# ==============================
coco_csv = r"D:\AVsenseinternship\avsens_object_detection\runs\detect\runs\detect\coco_train\results.csv"
coco_df = pd.read_csv(coco_csv)

print("\n=== COCO TRAINING METRICS (Per Epoch) ===")
print("Precision per epoch:")
print(coco_df["metrics/precision(B)"])

print("\nRecall per epoch:")
print(coco_df["metrics/recall(B)"])


# ==============================
# OPEN IMAGES RESULTS
# ==============================
oi_csv = r"D:\AVsenseinternship\avsens_object_detection\runs\detect\runs\detect\openimages_train\results.csv"
oi_df = pd.read_csv(oi_csv)

print("\n=== OPEN IMAGES TRAINING METRICS (Per Epoch) ===")
print("Precision per epoch:")
print(oi_df["metrics/precision(B)"])

print("\nRecall per epoch:")
print(oi_df["metrics/recall(B)"])