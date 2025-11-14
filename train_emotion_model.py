import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
import joblib
import matplotlib.pyplot as plt

# === 1. Load and Merge Data ===
folder = "emotion_data"
files = [f for f in os.listdir(folder) if f.endswith(".csv")]

dataframes = []
for f in files:
    df = pd.read_csv(os.path.join(folder, f))
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)
print(f"✅ Loaded {len(data)} samples from {len(files)} emotion files")

# === 2. Check for Missing Columns (Backward Compatibility) ===
expected_cols = ["leftEAR", "rightEAR", "MAR", "leftBrowDist", "rightBrowDist", "tilt"]
for col in expected_cols:
    if col not in data.columns:
        data[col] = 0.0  # placeholder for missing features
        print(f"⚠️ Missing feature '{col}' — filled with 0.0")

# === 3. Prepare Features and Labels ===
X = data[expected_cols].values
y = data["emotion"].values

# Label encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
n_classes = len(encoder.classes_)

# === 4. Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 5. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
)

# === 6. Build Neural Network Model ===
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === 7. Callbacks ===
checkpoint_cb = callbacks.ModelCheckpoint("best_emotion_model.h5",
                                          monitor='val_accuracy',
                                          save_best_only=True,
                                          verbose=1)
earlystop_cb = callbacks.EarlyStopping(monitor='val_loss',
                                       patience=8,
                                       restore_best_weights=True,
                                       verbose=1)

# === 8. Train ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=32,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=1
)

# === 9. Evaluate ===
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Final Accuracy: {acc*100:.2f}%")

# === 10. Save Model and Preprocessors ===
model.save("emotion_model.h5")
np.save("emotion_classes.npy", encoder.classes_)
joblib.dump(scaler, "scaler.pkl")

print("💾 Saved:")
print("- Model → emotion_model.h5")
print("- Classes → emotion_classes.npy")
print("- Scaler → scaler.pkl")

# === 11. Optional: Plot Training Curves ===
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Emotion Recognition Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()