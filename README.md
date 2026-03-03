# Pedestrian and Car Detection using YOLOv8  

### AVSens – AI/ML Systems Intern Selection Task

---

## 📌 Project Overview

This project was developed as part of the **AI/ML Systems Intern selection process conducted by AVSens**.  

The objective of the task was to design, train, evaluate, and analyze an object detection system capable of identifying **pedestrians and cars** in real-world street scenarios.

The project focuses not only on training a YOLO-based model, but also on **understanding model behavior**, **analyzing performance metrics**, **studying failure cases**, and **identifying improvement opportunities**—key aspects expected in real-world AI/ML systems.

---

## 🎯 Objectives

- Build an object detection system for **pedestrian and car detection**
- Train and compare models using **multiple datasets**
- Evaluate model performance using **precision, recall, and mAP**
- Perform **failure case analysis** and **confidence-based evaluation**
- Identify limitations and propose **practical improvements**

---

## 📂 Datasets Used

Three datasets were used for comparison:

### 1. COCO 2017 (Person & Car Subset)
- Generic dataset with diverse scenarios
- `person` class includes pedestrians, drivers, cyclists, etc.

### 2. Open Images V7 (Person & Car Subset)
- Large-scale dataset with varied annotation quality
- Used to analyze generalization behavior

### 3. Custom Pedestrian Dataset (Key Contribution)
- Manually curated and annotated dataset
- **Only pedestrians on foot were annotated**
- Humans inside vehicles, cyclists, and skateboarders were excluded
- Designed to reduce semantic ambiguity

---

## 🧠 Model Architecture

- **Model**: YOLOv8 (Ultralytics)
- **Input Size**: 640 × 640
- **Training Epochs**: 30
- **Frameworks Used**:
  - PyTorch
  - Ultralytics YOLO
  - FiftyOne (dataset preparation)
  - Python

---

## 📊 Evaluation Metrics

Models were evaluated using standard object detection metrics:

- **Precision**
- **Recall**
- **mAP@50**
- **mAP@50–95**

Each model was evaluated on its respective validation set to ensure fair comparison.

---

## 📈 Performance Summary

| Dataset | Precision | Recall | mAP@50 | mAP@50–95 |
|----------|----------|--------|--------|-----------|
| COCO | 0.646 | 0.491 | 0.547 | 0.311 |
| Open Images | 0.464 | 0.423 | 0.401 | 0.271 |
| Custom Pedestrian | **0.704** | **0.738** | **0.769** | **0.433** |

The custom pedestrian-specific dataset achieved the best performance across all metrics, highlighting the importance of task-specific annotation.

---

## 🔍 Failure Case Analysis

Observed failure cases include:
- False positives where humans inside vehicles are detected as pedestrians
- Missed pedestrians in crowded or occluded scenes
- Confidence-related errors at low thresholds

These failures were analyzed to understand model limitations and identify improvement opportunities.

---

## 📉 Confidence & Threshold Analysis

- Lower confidence thresholds increased recall but introduced false positives
- Higher thresholds reduced false detections but caused missed pedestrians
- Confidence-bin analysis was performed to study detection reliability

---

## 🛠 Improvements Attempted

- Custom pedestrian-only annotation strategy
- Dataset comparison across three sources
- Confidence threshold tuning
- Failure case categorization

---

## 🚀 Future Improvements

- Expand the custom dataset size
- Introduce separate classes for cyclists and vehicle occupants
- Explore contextual and temporal modeling
- Deploy the model for real-time inference

---

## 📌 Conclusion

This project demonstrates that **dataset semantics and annotation strategy** play a critical role in object detection performance. A smaller, well-defined dataset can outperform larger generic datasets when aligned closely with the task requirements.

---

## 🔗 Author

**Anush C Rao**  
AI/ML Engineering  
NMAM Institute of Technology  
