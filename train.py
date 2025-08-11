# train.py
import os
import numpy as np
from PIL import Image
from keras.api import Sequential
from keras.api.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.api.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from class_labels import classs  # Import dictionary

# Parameters
classes = len(classs)
img_size = (30, 30)

print("Total classes detected:", classes)
cur_path = os.getcwd()

# Load dataset
data, labels = [], []
print("Obtaining Images & their Labels ...........")
for i in range(classes):
    folder_path = os.path.join(cur_path, 'DataSets/TRAIN', str(i))
    for img_name in os.listdir(folder_path):
        try:
            image = Image.open(os.path.join(folder_path, img_name))
            image = image.resize(img_size).convert('RGB')
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            continue

print("Dataset loaded successfully!")

# Convert & normalize
data = np.array(data) / 255.0
labels = np.array(labels)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, shuffle=True
)

# One-hot encode labels
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Model definition
model = Sequential([
    Input(shape=img_size + (3,)),
    Conv2D(32, (5, 5), activation='relu'),
    Conv2D(32, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled. Training...")

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_test, y_test)
)

# Save model
model.save("myModel.h5")
print("Model saved as myModel.h5")

# Accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Accuracy.png")

# Loss plot
plt.figure()
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("LOSS.png")

print("Training completed & plots saved.")
