import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

X = []
y = []

for file in os.listdir("dataset"):
    if not file.endswith(".csv"):
        continue

    gesture = file.split(".")[0]
    df = pd.read_csv(f"dataset/{file}", header=None)

    X.extend(df.values)
    y.extend([gesture] * len(df))

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y = le.fit_transform(y)

X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(10,1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20)

model.save("model/gesture_cnn.h5")
print("Model Saved")
