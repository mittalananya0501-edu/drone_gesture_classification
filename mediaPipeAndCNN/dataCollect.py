import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

sample_count = 0
MAX_SAMPLES = 300


gesture_name = input("Enter gesture name: ")
save_path = f"dataset/{gesture_name}.csv"

if not os.path.exists("dataset"):
    os.makedirs("dataset")

file = open(save_path, mode='a', newline='')
writer = csv.writer(file)

fingertips = [4, 8, 12, 16, 20]

print("Press 's' to save data")

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

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            key = cv2.waitKey(1)
            if key == ord('s'):
               writer.writerow(data)
               sample_count += 1
               print(f"Saved: {sample_count}")

    if sample_count >= MAX_SAMPLES:
        print("Collected 300 samples!")
        break

    cv2.putText(frame, f"Samples: {sample_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2)


    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
file.close()
