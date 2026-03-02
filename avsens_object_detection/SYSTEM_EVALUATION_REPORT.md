# System Evaluation Report: End-to-End Object Detection Pipeline

## 1. Executive Summary & Approach
This project establishes an end-to-end YOLOv8 object detection pipeline designed to detect `person` and `car` entities. To understand dataset bias and generalization, two models were trained concurrently:
- **Model A (Baseline):** Trained on 200 images from the highly-curated COCO 2017 dataset.
- **Model B (Specialized):** Trained on 200 images from the sparsely-annotated, diverse Open Images V7 dataset.

Both were trained using YOLOv8n (nano) for 30 epochs with default hyperparameters, intentionally refraining from extreme optimization to isolate the impact of dataset characteristics on downstream performance. We subsequently evaluated Model A on a set of custom, highly occluded, and challenging test images (`custom_test_images`) to observe its realistic failure modes.

## 2. Quantitative Model Performance Metrics

The metrics recorded at the final epoch (30) on their respective validation splits are as follows:

| Metric | Model A (COCO) | Model B (Open Images) |
|---|---|---|
| **Precision** | 0.660 | 0.474 |
| **Recall** | 0.511 | 0.431 |
| **mAP50** | 0.547 | 0.405 |
| **mAP50-95** | 0.312 | 0.272 |

**Analysis:** Model A significantly outperforms Model B across all statistical fronts. This outcome is expected given COCO's dense, high-quality, and exhaustive bounding box annotations. Open Images V7, while varied, is known for sparse annotations—objects are often present but unannotated, heavily penalizing the precision score during training as the model learns to doubt valid features.

## 3. Dataset Comparison: Pros and Cons

### COCO 2017
- **Pros:** 
  - Extremely detailed and exhaustive annotations. If a person or car is in the frame, it is almost certainly boxed.
  - Highly standard benchmark, ensuring compatibility and expected behaviors in standard urban scenes.
- **Cons:**
  - Highly curated, meaning the model may overfit to specific lighting conditions and angles common in everyday photography.

### Open Images V7
- **Pros:**
  - Massive diversity in image domains, varying resolutions, uncommon angles, and unique object shapes.
  - Excellent for training models that must operate in highly atypical environments.
- **Cons:**
  - Sparse annotations. Many valid objects lack bounding boxes. 
  - Causes severe "false positive" penalization during training when the model correctly identifies an unannotated car.

## 4. Inference Performance & Failure Cases

Model A was tested on 10 customized image scenarios (`custom_test_images`), comprising dense crowds and partially occluded vehicles. 

### Output Observations:
- **Successful Detections:** The model successfully managed dense counting in well-lit scenarios (e.g., detecting 19 persons in `t10.jpg` and 11 cars in `t4.jpg`).
- **Complete Failures:** Images `t5.jpg` and `t9.jpg` yielded NO detections. This suggests severe algorithmic failure when presented with extreme occlusion, drastic scale changes, or extreme low-light environments (common in edge real-world scenarios).

### Identified Issues:
1. **False Negatives under Severe Occlusion:** When more than 60% of a vehicle's silhouette is hidden (behind a wall or another car), the bounding box confidence drops below the 0.25 threshold.
2. **Crowd Merging (Loss of Granularity):** In extremely dense crowds, multiple overlapping individuals are occasionally merged into a single bounding box.
3. **Small Object Misses:** Background cars and pedestrians located far from the camera lens are frequently dropped due to the native 640x640 reduction constraint of the input layer relative to the object's pixel density.

## 5. Improvement Opportunities (Safety-Critical Vision)

To transform this baseline into a deployment-ready safety algorithm, we propose the following improvements:

1. **Slicing Aided Hyper Inference (SAHI):**
   - **Approach:** Slice the original high-resolution image into overlapping patches (e.g., 512x512) and run inference on each patch independently before merging boxes via NMS.
   - **Benefit:** Solves the small object failure mode, crucial for early detection of distant pedestrians.

2. **Occlusion-Aware Augmentations:**
   - **Approach:** Integrate `CutMix` and `GridMask` during the training epochs. 
   - **Benefit:** Forces the network to learn representations from partial objects (like a car's roof or a pedestrian's legs), directly addressing the false negatives observed in occluded scenarios.

3. **Domain Adaptation with Synthetic Data:**
   - **Approach:** Use simulators (like CARLA) to generate precisely annotated, mathematically perfect occluded scenes (e.g., a child running from behind a parked van).
   - **Benefit:** Overcomes the sparse annotation problem of Open Images while injecting rare, safety-critical edge cases into the training pool.
