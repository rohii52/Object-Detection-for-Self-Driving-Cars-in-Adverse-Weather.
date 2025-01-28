# Object Detection for Self-Driving Cars in Adverse Weather

## Overview
This repository contains research on **Object Detection in Adverse Weather Conditions** for **self-driving cars** using **deep learning models**. The goal of this study is to compare the **Region-Based Convolutional Neural Networks (RCNN)** and **You Only Look Once (YOLOv5)** algorithms based on their **Mean Average Precision (mAP)** and **Intersection Over Union (IOU)** values.

The dataset used is the **DAWN (Detection in Adverse Weather Nature) dataset**, which consists of **1000 real-traffic images** categorized into four weather conditions: **fog, snow, rain, and sand**.

---

## Research Motivation
Autonomous driving systems rely on **vision-based sensors (cameras & LiDAR)** to perceive their surroundings. However, in extreme weather conditions such as **fog, rain, snow, and sandstorms**, these sensors suffer from distortions, leading to reduced object detection accuracy. This study aims to:
- Improve **real-time object detection** in self-driving cars.
- Evaluate the performance of **RCNN vs. YOLOv5** in adverse weather conditions.
- Enhance safety by optimizing deep learning models for autonomous vehicles.

---

## Datasets
The dataset used in this research is the **DAWN dataset**, which contains images labeled with object bounding boxes for vehicle detection under various weather conditions. The dataset is divided into the following categories:
- **Fog**
- **Rain**
- **Snow**
- **Sandstorm**


---

## Methodology
### **1. Data Preprocessing & Augmentation**
- **Resizing all images to 256x256** for uniform training.
- **Scaling bounding box coordinates** accordingly.
- **Horizontal flipping augmentation** to improve model generalization.

### **2. Object Detection Models**
#### **(a) Region-Based Convolutional Neural Network (RCNN)**
- Uses **Selective Search** to extract region proposals.
- Employs **VGG16 or ResNet50** as the feature extractor.
- Outputs class labels and bounding box regression for each proposal.

#### **(b) You Only Look Once (YOLOv5)**
- **Real-time object detection model** that divides images into grids.
- Uses **CSPDarknet** as the backbone and **Path Aggregation Network (PANet)** for enhanced feature propagation.
- Generates multi-scale predictions for **small, medium, and large objects**.

### **3. Model Training**
- **Loss Function**: Sum of classification and regression loss.
- **Optimizer**: Adam Optimizer.
- **Evaluation Metrics**:
  - **mAP (Mean Average Precision)**
  - **IOU (Intersection Over Union)**
  - **Inference Speed (FPS - Frames Per Second)**

---

## Results
### **RCNN Performance**
- **All Classes**:
  - Train **mAP**: 0.231
  - Test **mAP**: 0.207
- **Binary Class (Car + Background)**:
  - Train **mAP**: 0.279
  - Test **mAP**: 0.216

### **YOLOv5 Performance**
- **Higher precision and faster inference speed** compared to RCNN.
- Handles **small and occluded objects better**.
- **More robust to foggy weather conditions**.

---

## Repository Structure
```
├── Dataset
│   ├── Fog (Images under foggy conditions)
│   ├── Rain (Images under rainy conditions)
│   ├── Snow (Images under snowy conditions)
│   ├── Sand (Images under sandstorm conditions)
│
├── Paperwork
│   ├── Thesis.pdf (Research Report)
│
├── References
│   ├── Papers on Object Detection in Adverse Weather
│
├── Source Code
│   ├── Code.ipynb (Jupyter Notebook for Training & Evaluation)
│
└── README.md (Project Documentation)
```

---

## Installation & Usage
### **1. Clone Repository**
```bash
git clone https://github.com/yourusername/Object-Detection-for-Self-Driving-Cars-in-Adverse-Weather.git
cd Object-Detection-for-Self-Driving-Cars-in-Adverse-Weather
```

### **2. Set Up the Environment**
```bash
pip install -r requirements.txt
```

### **3. Run the Training Script**
```python
python train.py --dataset Dataset/Fog --model yolov5
```

### **4. Run Object Detection**
```python
test.py --image test_image.jpg --model yolov5
```

---

## Conclusion
This study demonstrates the performance comparison of **RCNN vs. YOLOv5** for **object detection in adverse weather conditions**. The results indicate:
- **YOLOv5 significantly outperforms RCNN** in terms of accuracy and speed.
- **Foggy conditions pose challenges** for both models, but **YOLOv5 adapts better**.
- **Future improvements** include data augmentation techniques and model optimizations for real-world deployment.

---

## Author
[Rohith Ganesan](https://github.com/rohi52)

---

## References
1. 3D Object Detection with SLS-Fusion Network in Foggy Weather Conditions.
2. Road Object Detection: A Comparative Study of Deep Learning-Based Algorithms.
3. Machine and Deep Learning Techniques for Daytime Fog Detection in Real Time.
4. Adaptive Model for Object Detection in Noisy and Fast-Varying Environment.

For further details, refer to **Thesis.pdf** in this repository.
