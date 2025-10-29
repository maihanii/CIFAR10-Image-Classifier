import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("🔹 Starting training phase for CIFAR-10 (TensorFlow version)...")

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"✅ Data prepared successfully:")
print(f"   x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"   x_test: {x_test.shape}, y_test: {y_test.shape}")

print("🧠 Building the CNN model...")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Training started... this might take a few minutes.")


callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("model/best_model.keras", save_best_only=True)
]

history = model.fit(
    x_train, y_train,
    epochs=25,  
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

print("✅ Training completed!")

model.save("model/cifar10_cnn.keras")
print("💾 Model saved successfully at model/cifar10_cnn.keras")

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/accuracy.png")
plt.close()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss.png")
plt.close()

print("📊 Saved training plots to results/ folder.")
print("🎉 Training phase finished successfully!")

