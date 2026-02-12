import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
import pickle

model = load_model("model/gesture_cnn.h5")

class_names = ["DOWN", "FLIP", "LAND", "LEFT", "RIGHT", "TAKEOFF", "UP"] 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(1)

fingertips = [4, 8, 12, 16, 20]

prev_gesture = None
start_time = None
HOLD_TIME = 5  

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            data = []
            for id in fingertips:
                lm = hand_landmarks.landmark[id]
                data.append(lm.x)
                data.append(lm.y)

            X = np.array(data).reshape(1,10,1)
            prediction = model.predict(X)
            gesture_index = np.argmax(prediction)
            gesture_name = class_names[gesture_index]


            if gesture_name == prev_gesture:
                if start_time and time.time() - start_time > HOLD_TIME:
                    print("Command Triggered:", gesture_name)
            else:
                prev_gesture = gesture_name
                start_time = time.time()

            cv2.putText(frame, gesture_name, (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
