import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("🔹 Starting EDA for CIFAR-10 (TensorFlow version)...")

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(f"✅ Dataset loaded successfully:")
print(f"  Train shape: {x_train.shape}")
print(f"  Test shape: {x_test.shape}")

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

print("🖼️ Generating sample images...")
fig, axes = plt.subplots(3, 5, figsize=(10,6))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(x_train))
    ax.imshow(x_train[idx])
    ax.set_title(classes[int(y_train[idx])])
    ax.axis("off")
plt.tight_layout()
plt.savefig("plots/sample_images.png")
plt.close()
print("✅ Saved sample_images.png")

print("📊 Generating class distribution plot...")
unique, counts = np.unique(y_train, return_counts=True)
sns.barplot(x=[classes[i] for i in unique], y=counts)
plt.title("Class Distribution in Training Set")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/class_distribution.png")
plt.close()
print("✅ Saved class_distribution.png")

print("🔍 Pixel value range:")
print(f"  min={x_train.min()}, max={x_train.max()}, dtype={x_train.dtype}")
with open("data/summary.txt", "w") as f:
    f.write("CIFAR-10 Dataset Summary (TensorFlow)\n")
    f.write(f"Train samples: {len(x_train)}\n")
    f.write(f"Test samples: {len(x_test)}\n")
    f.write("Classes:\n")
    for c in classes:
        f.write(f" - {c}\n")
print("✅ Saved data/summary.txt")

print("🎉 EDA finished successfully!")
