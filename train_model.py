import os
import json
import numpy as np#numerical clauclation
import matplotlib.pyplot as plt#plotting graphs
from sklearn.metrics import classification_report, confusion_matrix#evalution metrices
from tensorflow.keras.preprocessing.image import ImageDataGenerator#building cnn model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ----------------------------
# CONFIGURATION
# ----------------------------
DATASET_DIR = "dataset"  # Change dataset folder name if required
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
MODEL_SAVE_PATH = "skin_model.h5"
CLASS_INDEX_FILE = "class_indices.json"#stores disease labels

# ----------------------------
# DATA PREPROCESSING
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2#function from tensorflow
)

train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,#load datset from folder
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)

# ----------------------------
# SAVE CLASS LABELS
# ----------------------------
with open(CLASS_INDEX_FILE, 'w') as f:
    json.dump(train_data.class_indices, f, indent=4)
print("Class indices saved to:", CLASS_INDEX_FILE)

# ----------------------------
# BUILD MODEL (MobileNetV2)
# ----------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(train_data.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# CALLBACKS
# ----------------------------
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)
earlystop = EarlyStopping(patience=5, restore_best_weights=True)

# ----------------------------
# TRAIN MODEL
# ----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

model.save(MODEL_SAVE_PATH)
print("Model Saved Successfully as:", MODEL_SAVE_PATH)

# ----------------------------
# EVALUATION
# ----------------------------
val_data.reset()
predictions = model.predict(val_data)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = val_data.classes
labels = list(train_data.class_indices.keys())

print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=labels))

# ----------------------------
# PLOT ACCURACY & LOSS
# ----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss")
plt.legend()

plt.show()
