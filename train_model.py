import numpy as np
import cv2
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 64
DATADIR  = "Data/Training"
CATEGORIES = ["male", "female"]   # 0 = male, 1 = female

# ── 1. Load images ──────────────────────────────────────────────────────────
training_data = []

for category in CATEGORIES:
    path      = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    files     = os.listdir(path)
    print(f"Loading {len(files)} images from {category}…")
    for img_name in files:
        try:
            img_array = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
        except Exception:
            pass

print(f"\nTotal images loaded: {len(training_data)}")

# ── 2. Shuffle & split ───────────────────────────────────────────────────────
random.shuffle(training_data)

X, y = [], []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
y = np.array(y)

# ── 3. Build CNN ─────────────────────────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ── 4. Train ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
]

history = model.fit(
    X, y,
    batch_size=32,
    epochs=15,
    validation_split=0.1,
    callbacks=callbacks
)

# ── 5. Save ───────────────────────────────────────────────────────────────────
model.save("gender_model.h5")
print("\nModel saved as gender_model.h5")

# Print final accuracy
val_acc = max(history.history['val_accuracy'])
print(f"Best validation accuracy: {val_acc*100:.2f}%")
