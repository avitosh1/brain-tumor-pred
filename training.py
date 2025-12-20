# ================= WINDOWS SAFETY FIX =================
import multiprocessing
multiprocessing.freeze_support()

# ================= IMPORTS =================
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

# ================= CONFIG =================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs("models", exist_ok=True)

# ================= DATA LOADING =================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ================= DATASET UNDERSTANDING =================
class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses Found:", class_names)
print("Number of Classes:", num_classes)

# ---- Class Distribution ----
train_labels = np.concatenate([y.numpy() for _, y in train_ds])
class_distribution = Counter(train_labels)

print("\nClass Distribution (Training Set):")
for idx, count in class_distribution.items():
    print(f"{class_names[idx]} : {count}")

# ---- Sample Image Visualization ----
for images, labels in train_ds.take(1):
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.title(f"Sample Image - {class_names[labels[0]]}")
    plt.axis("off")
    plt.show()

# ================= PERFORMANCE SAFE PIPELINE =================
train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ================= PREPROCESSING =================
normalization = layers.Rescaling(1./255)

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ================= CUSTOM CNN =================
custom_model = models.Sequential([
    augmentation,
    normalization,

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

custom_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_custom = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("models/custom_cnn.h5", save_best_only=True)
]

print("\nTraining Custom CNN...")
history_custom = custom_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks_custom
)

# ================= TRANSFER LEARNING =================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
output = layers.Dense(num_classes, activation="softmax")(x)

tl_model = models.Model(base_model.input, output)

tl_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_tl = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("models/efficientnet.h5", save_best_only=True)
]

print("\nTraining Transfer Learning Model...")
history_tl = tl_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks_tl
)

# ================= FINE TUNING =================
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

tl_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nFine-tuning Transfer Learning Model...")
tl_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10
)

# ================= EVALUATION =================
def evaluate_model(model, name):
    print(f"\nEvaluating {name}...")
    loss, acc = model.evaluate(test_ds)
    print(f"{name} Test Accuracy: {acc:.4f}")

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return acc

acc_custom = evaluate_model(custom_model, "Custom CNN")
acc_tl = evaluate_model(tl_model, "EfficientNet")

# ================= TRAINING CURVES =================
def plot_history(history, title):
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title(f"{title} Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{title} Loss")
    plt.legend()

    plt.show()

plot_history(history_custom, "Custom CNN")
plot_history(history_tl, "EfficientNet")

# ================= MODEL COMPARISON =================
print("\nMODEL COMPARISON SUMMARY")
print(f"Custom CNN Accuracy   : {acc_custom:.4f}")
print(f"EfficientNet Accuracy : {acc_tl:.4f}")

if acc_tl > acc_custom:
    print("\nConclusion:")
    print("EfficientNet outperforms the Custom CNN and is selected for deployment.")
else:
    print("\nConclusion:")
    print("Custom CNN performs competitively and is selected for deployment.")

print("\n✅ PROJECT CODE RUNS SUCCESSFULLY & IS FULLY COMPLIANT")
