# 🧠 Brain Tumor Detection using Deep Learning

A deep learning-based medical imaging project that classifies brain MRI scans to detect the presence of a tumor using a **fine-tuned MobileNetV2 CNN model**. It also integrates **Grad-CAM** visualizations to highlight the tumor regions and includes a simple **Streamlit** web app for real-time diagnosis.

---

## 🚀 Project Overview

Brain tumors are life-threatening conditions that require early detection for effective treatment.  
This project leverages **transfer learning** with MobileNetV2 to classify brain MRI images into **tumor** and **non-tumor** categories.  
The trained model is deployed through a web app that allows users to upload MRI images and get real-time predictions with **heatmap-based explainability** using Grad-CAM.

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit (for interactive image upload and prediction interface)  
- **Backend**: Python, TensorFlow/Keras  
- **Model**: MobileNetV2 (transfer learning)  
- **Visualization**: Grad-CAM for tumor region explainability  
- **Data**: Public Brain Tumor MRI datasets (e.g., Kaggle)

---

## 📁 Dataset & Model Info

- Two classes:
  - `Tumor`
  - `Normal`
- Image preprocessing includes:
  - Resize to 224×224
  - Normalization
  - Data augmentation (rotation, zoom, flipping)
- **Model**: Fine-tuned MobileNetV2 with added dense classification layers
- **Explainability**: Grad-CAM highlights the region where the model focuses

⚠️ **Note**:  
- The full dataset and trained `.h5` model file are **not included** in this repository due to size limitations.  
- Sample images and key code for preprocessing, training, and visualization are included.

🔗 Example dataset used:  
- [Kaggle – Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## 📊 Model Performance

- **Accuracy**: 94.8% on validation set  
- **Precision**: 93.2%  
- **Recall**: 95.5%  
- Lightweight model, deployable on local machines in real-time  
- Visual heatmaps help in clinical interpretability (Grad-CAM)

---

## 🛠 How to Run the Project

Follow these steps to run the project locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/OmiVibes/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection

2. **Install the dependencies**
   - Run the following command to install the required Python libraries:
     ```bash
     pip install -r requirements.txt
     ```
3. **Run the Flask application**
   - Start the Flask server using:
     ```bash
     python app.py
     ```
4. **Access the application**
   - Open your browser and visit:
     ```
     http://localhost:5000/
     ```

---

## 👨‍💻 Author

**Om Shinde**  
📧 [omshinde1819@gmail.com](mailto:omshinde1819@gmail.com)  
🌐 [GitHub – OmiVibes](https://github.com/OmiVibes)

---     