import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model/cifar10_cnn.keras")

# CIFAR-10 class labels
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Create main window
window = tk.Tk()
window.title("CIFAR-10 Image Classifier")
window.geometry("500x500")
window.config(bg="#f2f2f2")

# Function to open and classify an image
def upload_and_classify():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if not file_path:
        return

    try:
        img = Image.open(file_path).convert("RGB").resize((128, 128))
    except Exception as e:
        result_label.config(text=f"Error loading image: {e}", fg="red")
        return

    tk_img = ImageTk.PhotoImage(img)
    panel.config(image=tk_img)
    panel.image = tk_img

    img_array = np.array(img.resize((32, 32))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"Prediction: {class_names[class_index]}\nConfidence: {confidence:.2f}%",
        fg="#333"
    )
title_label = tk.Label(
    window, text="CIFAR-10 Image Classifier",
    font=("Arial", 18, "bold"), bg="#f2f2f2", fg="#333"
)
title_label.pack(pady=15)

upload_btn = tk.Button(
    window, text="Upload & Classify Image",
    command=upload_and_classify,
    font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5
)
upload_btn.pack(pady=10)

panel = tk.Label(window, bg="#ddd", width=256, height=256)
panel.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 14), bg="#f2f2f2", fg="#333")
result_label.pack(pady=10)

exit_btn = tk.Button(
    window, text="Exit",
    command=window.destroy,
    font=("Arial", 12), bg="#f44336", fg="white", padx=10, pady=5
)
exit_btn.pack(pady=10)

window.mainloop()
