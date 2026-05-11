import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(file_path):

    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        ).T,
        axis=0
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=audio,
            sr=sample_rate
        ).T,
        axis=0
    )

    mel = np.mean(
        librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate
        ).T,
        axis=0
    )

    feature = np.hstack((mfcc, chroma, mel))

    return feature


# -----------------------------
# Load Dataset
# -----------------------------
dataset_path = "dataset"

features = []
labels = []

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            emotion = file.split("-")[2]

            feature = extract_features(file_path)

            features.append(feature)
            labels.append(emotion)


# -----------------------------
# Convert to arrays
# -----------------------------
X = np.array(features)
y = np.array(labels)


# -----------------------------
# Normalize Features
# -----------------------------
scaler = StandardScaler()

X = scaler.fit_transform(X)


# -----------------------------
# Encode Labels
# -----------------------------
encoder = LabelEncoder()

y = encoder.fit_transform(y)

y = to_categorical(y)

emotion_labels = encoder.classes_


# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Reshape for LSTM
# -----------------------------
X_train = np.expand_dims(X_train, axis=1)

X_test = np.expand_dims(X_test, axis=1)


# -----------------------------
# Build Improved LSTM Model
# -----------------------------
model = Sequential()

model.add(
    LSTM(
        256,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dropout(0.3))

model.add(LSTM(128))

model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

model.add(Dense(y.shape[1], activation='softmax'))


# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# -----------------------------
# Train Model
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# -----------------------------
# Evaluate Model
# -----------------------------
predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)

print("\nModel Accuracy:", accuracy)


# -----------------------------
# Emotion Prediction
# -----------------------------
def predict_emotion(audio_path):

    audio, sample_rate = librosa.load(
        audio_path,
        duration=3,
        offset=0.5
    )

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        ).T,
        axis=0
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=audio,
            sr=sample_rate
        ).T,
        axis=0
    )

    mel = np.mean(
        librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate
        ).T,
        axis=0
    )

    feature = np.hstack((mfcc, chroma, mel))

    feature = scaler.transform([feature])

    feature = np.expand_dims(feature, axis=1)

    prediction = model.predict(feature)

    predicted_index = np.argmax(prediction)

    emotion_dict = {
        "01": "Neutral",
        "02": "Calm",
        "03": "Happy",
        "04": "Sad",
        "05": "Angry",
        "06": "Fearful",
        "07": "Disgust",
        "08": "Surprised"
    }

    emotion = emotion_dict.get(
        emotion_labels[predicted_index],
        "Unknown"
    )

    print("\nPredicted Emotion:", emotion)


# -----------------------------
# Ask user for audio file
# -----------------------------
audio_file = input("\nEnter audio file path: ")

if os.path.exists(audio_file):

    predict_emotion(audio_file)

else:

    print("Audio file not found. Please check the path.")