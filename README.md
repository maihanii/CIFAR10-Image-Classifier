# 🧠 CIFAR-10 Image Classifier (TensorFlow + Tkinter GUI)

A complete deep learning project that trains a Convolutional Neural Network (CNN) on the **CIFAR-10 dataset**,  
evaluates it, and provides a graphical interface (GUI) for image classification.

---

## 📖 Overview

This project demonstrates a full deep learning workflow:
1. **Data Exploration (EDA)** – Visualizing sample images and class distribution.  
2. **Model Training** – Building and training a CNN using TensorFlow/Keras.  
3. **Evaluation** – Generating metrics, confusion matrix, and performance plots.  
4. **GUI Interface** – Allowing users to upload and classify custom images.

---

## 📂 Project Structure



DeepLearning/
├── data/
│ └── summary.txt
│
├── model/
│ ├── cifar10_cnn.keras
│ └── best_model.keras
│
├── plots/
│ ├── sample_images.png
│ └── class_distribution.png
│
├── results/
│ ├── accuracy.png
│ ├── loss.png
│ ├── confusion_matrix.png
│ ├── classification_report.txt
│ └── sample_predictions.png
│
├── src/
│ ├── eda.py
│ ├── train.py
│ ├── evaluate.py
│ ├── gui.py
│ └── utils.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── main.py


---

## ⚙️ Setup Instructions

### 1️⃣ Create a Virtual Environment

```bash
python -m venv venv


Activate it:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

2️⃣ Install Dependencies
pip install -r requirements.txt

🚀 How to Run the Project
▶️ Step 1: Explore the Data (EDA)
python src/eda.py

▶️ Step 2: Train the CNN Model
python src/train.py

▶️ Step 3: Evaluate Model Performance
python src/evaluate.py

▶️ Step 4: Launch the GUI App
python src/gui.py

📊 Output Files
Folder	Contents
plots/	EDA images (sample images, class distribution)
results/	Accuracy & loss plots, confusion matrix, classification report
model/	Trained CNN model files (cifar10_cnn.keras, best_model.keras)
data/	Dataset summary info
🧩 Technologies Used

Python 3.10+

TensorFlow / Keras

NumPy

Matplotlib / Seaborn

scikit-learn

Pillow

Tkinter (for GUI)

🖼️ GUI Preview

Upload any image (e.g., airplane, cat, truck...) and the trained model will predict its class with confidence!

+------------------------------------------+
|  [Upload & Classify Image]               |
|                                          |
|     [ Image Preview ]                    |
|                                          |
|  Prediction: airplane                    |
|  Confidence: 93.45%                      |
+------------------------------------------+

📈 Example Results

Accuracy: ~75–80% (depending on training time)

Loss Curve: smooth convergence after ~15 epochs

Confusion Matrix: well-balanced performance across 10 classes

👨‍💻 Author

Developed by mai hani ramadan 
Built with ❤️ using TensorFlow, Keras, and Tkinter