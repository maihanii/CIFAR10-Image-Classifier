import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("🔹 Starting model evaluation...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255.0
y_test = tf.keras.utils.to_categorical(y_test, 10)

model_path = "model/cifar10_cnn.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model file not found! Run train.py first.")

model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"📊 Test Accuracy: {accuracy*100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")

y_pred = model.predict(x_test)
y_true_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - CIFAR-10')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("results/confusion_matrix.png")
plt.close()
print("✅ Saved confusion_matrix.png")

report = classification_report(y_true_classes, y_pred_classes)
with open("results/classification_report.txt", "w") as f:
    f.write(report)
print("✅ Saved classification_report.txt")

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

plt.figure(figsize=(12, 6))
for i in range(10):
    idx = np.random.randint(0, len(x_test))
    img = x_test[idx]
    true_label = class_names[y_true_classes[idx]]
    pred_label = class_names[y_pred_classes[idx]]
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    color = "green" if true_label == pred_label else "red"
    plt.title(f"T:{true_label}\nP:{pred_label}", color=color)
    plt.axis("off")
plt.tight_layout()
plt.savefig("results/sample_predictions.png")
plt.close()
print("✅ Saved sample_predictions.png")

print("🎉 Evaluation finished successfully!")
