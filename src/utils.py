import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10

def ensure_directories():
    """Ensure folders (data, model, plots) exist."""
    for folder in ["data", "model", "plots"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    print("✅ Folders checked or created successfully.")

def load_cifar10_data():
    """Load CIFAR-10 dataset and normalize images."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(f"✅ CIFAR-10 loaded: {len(x_train)} train, {len(x_test)} test samples.")
    return (x_train, y_train), (x_test, y_test)

def plot_sample_images(x, y, class_names, save_path="plots/sample_images.png"):
    """Plot sample images from the dataset."""
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i])
        plt.title(class_names[int(y[i])])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved sample images to {save_path}")

def plot_class_distribution(y, class_names, save_path="plots/class_distribution.png"):
    """Plot how many images per class."""
    counts = [np.sum(y == i) for i in range(len(class_names))]
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, counts, color="skyblue")
    plt.xticks(rotation=45)
    plt.title("CIFAR-10 Class Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved class distribution plot to {save_path}")
