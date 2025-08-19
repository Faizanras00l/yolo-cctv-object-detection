# 🚦 CCTV Object Detection with YOLO 🚦

Welcome to a robust, production-ready solution for object detection on CCTV footage using the powerful YOLO architecture!  
This repository enables seamless inference both locally and on Google Colab, making advanced surveillance analytics accessible to everyone.

---

## 📋 Table of Contents

- [✨ Project Overview](#-project-overview)
- [⚡️ Setup](#-setup)
  - [Google Colab](#google-colab)
  - [Local Inference](#local-inference)
- [🚀 Usage](#-usage)
  - [Colab Interface](#colab-interface)
  - [Local Interface](#local-interface)
- [📊 Results & Evaluation](#-results--evaluation)
- [📁 Project Structure](#-project-structure)
- [📝 License](#-license)

---

## ✨ Project Overview

Leverage the [YOLO](https://github.com/ultralytics/ultralytics) architecture for real-time object detection on CCTV images and videos.  
**Key detection capabilities include:**  
- 🚗 Cars  
- 🚚 Large cars  
- 🚶 Persons  
- 🌊 Flood  
- 💥 Accidents  
...and more, depending on your trained model weights!

This repository provides:
- Pre-trained weights
- Ready-to-use scripts for both local and Colab inference
- Comprehensive evaluation results and visualizations

> ⚠️ **Note:** Please verify that your model and weights are trained for your specific detection requirements.

---

## ⚡️ Setup

### Google Colab

1. Open the `inference_colab_(CCTV).ipynb` notebook in Google Colab.
2. Upload your trained weights (e.g., `cctv.pt`) and the image/video you want to analyze.
3. Follow the notebook instructions to run inference and visualize results.

### Local Inference

1. Clone this repository and navigate to the project directory.
2. Ensure you have Python 3.8+ installed.
3. Install dependencies:
    ```
    pip install ultralytics opencv-python torch
    ```
4. Place your trained weights (e.g., `cctv.pt`) in the project directory.
5. Place the image or video you want to test in the project directory.

---

## 🚀 Usage

### Colab Interface

- Open `inference_colab_(CCTV).ipynb` in Colab.
- The notebook will:
  - Install required packages
  - Prompt you to upload your model weights and test image/video
  - Run inference and display results inline (images or video previews)

### Local Interface

- Edit `inference_Local_(CCTV).py` to set your model path and input file.
- Run the script:
    ```
    python inference_Local_(CCTV).py
    ```
- The script will:
  - Load your trained model
  - Run inference on the specified image or video
  - Display the result (for images) or print the output file path (for videos)

---

## 📊 Results & Evaluation

Explore detailed evaluation results and metric curves in the `Results` folder:
- `BoxP_curve.png`: Precision curve
- `BoxR_curve.png`: Recall curve
- `BoxF1_curve.png`: F1 score curve
- `confusion_matrix.png` and `confusion_matrix_normalized.png`: Confusion matrices
- `results.csv`: Tabular results for each class and overall metrics
- Visual samples of model predictions and ground truth labels (e.g., `val_batch0_labels.jpg`, `val_batch0_pred.jpg`)

---

## 📁 Project Structure

```
.
├── inference_colab_(CCTV).ipynb      # Colab notebook for inference
├── inference_Local_(CCTV).py         # Local Python script for inference
├── weights/                          # Folder containing trained model weights
│   ├── best_Weights.pt
│   ├── best_Weights(2.0).pt
│   └── last_Weights.pt
├── Results/                          # Evaluation results and visualizations
│   ├── args.yaml                     # Training arguments and configuration
│   ├── BoxF1_curve.png               # F1 score curve for bounding boxes
│   ├── BoxP_curve.png                # Precision curve for bounding boxes
│   ├── BoxPR_curve.png               # Precision-Recall curve
│   ├── BoxR_curve.png                # Recall curve for bounding boxes
│   ├── confusion_matrix_normalized.png # Normalized confusion matrix
│   ├── confusion_matrix.png          # Confusion matrix
│   ├── labels_correlogram.jpg        # Label correlation visualization
│   ├── labels.jpg                    # Label distribution
│   ├── results.csv                   # Detailed results per class
│   ├── results.png                   # Summary results visualization
│   ├── train_batch0.jpg              # Example training batch images
│   ├── train_batch1.jpg
│   ├── train_batch2.jpg
│   ├── train_batch1080.jpg
│   ├── train_batch1081.jpg
│   ├── train_batch1082.jpg
│   ├── val_batch0_labels.jpg         # Validation batch ground truth
│   └── val_batch0_pred.jpg           # Validation batch predictions
```
- **inference_colab_(CCTV).ipynb**: Jupyter notebook for running inference in Google Colab  
- **inference_Local_(CCTV).py**: Python script for running inference locally  
- **weights/**: Contains different versions of trained model weights  
- **Results/**: Contains all evaluation metrics, visualizations, and sample outputs  

---

## 📝 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

**💬 For any questions or issues, please open an issue on this repository.**