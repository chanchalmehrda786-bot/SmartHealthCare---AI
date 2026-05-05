import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import os

# ── Create output folders if they don't exist ──────────────────────────────
os.makedirs("Model", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
train = pd.read_csv("data/Training.csv")

# Remove unwanted column
train = train.drop(columns=["Unnamed: 133"], errors='ignore')

# Split features and target
X = train.iloc[:, :-1]
y = train.iloc[:, -1]

# Convert to numeric and clean
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)
X = X.astype(np.float32)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save encoder using pickle
with open("Model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"✅ Label encoder saved — {len(le.classes_)} disease classes")

# ── Build model (matches architecture shown in apps.py About page) ─────────
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),

    Dense(len(np.unique(y_encoded)), activation='softmax')
])

# ── Compile ────────────────────────────────────────────────────────────────
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ──────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

# ── Train ──────────────────────────────────────────────────────────────────
history = model.fit(
    X, y_encoded,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# ── Save model ─────────────────────────────────────────────────────────────
model.save("Model/disease_model.h5")
print("✅ Model trained and saved to Model/disease_model.h5")

# ── Quick training summary ─────────────────────────────────────────────────
final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"   Final Training Accuracy : {final_acc * 100:.2f}%")
print(f"   Final Validation Accuracy: {final_val_acc * 100:.2f}%")