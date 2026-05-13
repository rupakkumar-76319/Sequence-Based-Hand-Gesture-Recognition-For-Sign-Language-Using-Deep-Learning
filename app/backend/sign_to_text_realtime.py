import cv2
import numpy as np
import torch
import pickle
import mediapipe as mp
from collections import deque

from models.Models.model_LSTM import SignLanguageLSTM
from models.Models.model_GRU import SignLanguageGRU
from models.Models.model_BiLSTM import SignLanguageBiLSTM
from models.Models.model_CNN import SignLanguageCNN
from models.Models.model_CNNGRU import CNN_GRU
from models.Models.model_CNNBiLSTM import CNN_BiLSTM


# SELECT MODEL

model_name = input("Choose model (lstm / gru / bilstm / cnn / cnn_gru / cnn_bilstm): ").lower()

BASE_MODEL_PATH = {
    "lstm": "models/Models/lstm_best.pth",
    "gru": "models/Models/gru_best.pth",
    "bilstm": "models/Models/bilstm_best.pth",
    "CNN": "models/Models/cnn_best.pth",
    "CNN_GRU": "models/Models/cnn_gru_best.pth",
    "CNN_BiLSTM": "models/Models/cnn_bilstm_best.pth",
}

if model_name == "lstm":
    model = SignLanguageLSTM()

elif model_name == "gru":
    model = SignLanguageGRU()

elif model_name == "bilstm":
    model = SignLanguageBiLSTM()

elif model_name == "cnn":
    model = SignLanguageCNN()

elif model_name == "cnn_gru":
    model = CNN_GRU()
    
elif model_name == "cnn_bilstm":
    model = CNN_BiLSTM()

else:
    raise ValueError("❌ Invalid model choice")

MODEL_PATH = BASE_MODEL_PATH[model_name]


# CONFIG

ENCODER_PATH = "data/objects/label_encoder.pkl"
SEQ_LENGTH = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL

model = model.to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(f"Loaded {model_name.upper()} model")

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

sequence = deque(maxlen=SEQ_LENGTH)
predictions = deque(maxlen=15)

current_label = ""
current_confidence = 0.0

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        kp = []
        for lm in hand_landmarks.landmark:
            kp.append([lm.x, lm.y, lm.z])  # ✅ correct

        kp = np.array(kp)

        # ✅ same normalization as training
        kp = kp - kp[0]

        keypoints = kp.flatten()
        sequence.append(keypoints)

    else:
        if len(sequence) > 0:
            sequence.append(sequence[-1])

    
    if len(sequence) == SEQ_LENGTH:

        input_data = np.array(sequence)
        input_data = np.expand_dims(input_data, axis=0)

        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)

            confidence, pred = torch.max(probs, dim=1)
            confidence = confidence.item()
            pred = pred.item()

        if confidence > 0.6:
            predictions.append(pred)

        if len(predictions) >= 10:
            final_pred = max(set(predictions), key=predictions.count)
            count = predictions.count(final_pred)

            if count >= 7:
                label = encoder.inverse_transform([final_pred])[0]

                current_label = label
                current_confidence = confidence

                predictions.clear()

    
    
    if current_label:
        cv2.putText(frame, f"{model_name.upper()} → {current_label}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 4)

        cv2.putText(frame, f"Conf: {current_confidence:.0%}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()