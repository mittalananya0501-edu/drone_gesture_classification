import cv2
import mediapipe as mp
# test change

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

current_gesture = "SHOW COMMAND"
gesture_buffer = [] 
last_raw_command = None  
BUFFER_SIZE = 15 

def get_gesture(lm_list):
    fingers = []
    fingers.append(1 if lm_list[4][1] > lm_list[3][1] else 0)
    for tip in [8, 12, 16, 20]:
        fingers.append(1 if lm_list[tip][2] < lm_list[tip - 2][2] else 0)

    if fingers == [1, 1, 1, 1, 1]: return "TAKEOFF"
    if fingers == [0, 0, 0, 0, 0]: return "LAND"
    if fingers == [0, 1, 0, 0, 0]: return "MOVE UP"
    if fingers == [0, 0, 0, 0, 1]: return "MOVE DOWN"
    if fingers == [1, 0, 0, 0, 0]: return "MOVE LEFT"
    if fingers == [0, 1, 1, 1, 1]: return "MOVE RIGHT"
    if fingers == [0, 1, 1, 0, 0]: return "FLIP"
    return "DEAD"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    raw_command = "NO HAND"

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm_list = [[id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])] 
                       for id, lm in enumerate(hand_lms.landmark)]
            
            raw_command = get_gesture(lm_list)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    if raw_command != last_raw_command:
        gesture_buffer = []
        last_raw_command = raw_command

    if raw_command != "NO HAND":
        gesture_buffer.append(raw_command)

    if len(gesture_buffer) > BUFFER_SIZE:
        gesture_buffer.pop(0)

    if len(gesture_buffer) == BUFFER_SIZE:
        current_gesture = gesture_buffer[0]
    elif raw_command == "NO HAND":
        current_gesture = "SHOW COMMAND"

    fill_width = int((len(gesture_buffer) / BUFFER_SIZE) * 200)
    cv2.rectangle(img, (20, 80), (220, 95), (50, 50, 50), -1) 
    cv2.rectangle(img, (20, 80), (20 + fill_width, 95), (0, 255, 0), -1) 

    cv2.putText(img, f"DRONE: {current_gesture}", (20, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 230), 5)

    cv2.imshow("Hand Gesture Drone Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()