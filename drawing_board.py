import mediapipe as mp
import numpy as np
import math
import cv2

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=4)

mp_drawing = mp.solutions.drawing_utils

# Capture video from your webcam.
cap = cv2.VideoCapture(0)

def check_pinch(thumb, finger_tip):
    if len(thumb):
        is_similar = all(
            math.isclose(a[0], b[0], abs_tol=0.035) and
            math.isclose(a[1], b[1], abs_tol=0.035) and
            math.isclose(a[2], b[2], abs_tol=0.035)
            for a, b in zip(thumb, finger_tip)
        )
        return is_similar
    return False

paths = []
current_path = []

while cap.isOpened():
    right_thumb = []
    right_finger_tip = []
    left_thumb = []
    left_finger_tip = []

    success, image = cap.read()
    h, w, _ = image.shape

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display.
    # Convert the BGR image to RGB.
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hand Positions
    results = hands.process(image)

    black_display = np.zeros((h, w, 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            mp_drawing.draw_landmarks(
                black_display, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            if hand_label == "Right":
                right_thumb.append((hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z))

                right_finger_tip.append((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z))
                
            if hand_label == "Left":
                left_thumb.append((hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z))

                left_finger_tip.append((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z))
    if not len(paths):
        cv2.putText(black_display, "Right Pinch to draw, Left Pinch to reset", (200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)
    
    if check_pinch(left_thumb, left_finger_tip):
        paths = []
        current_path = []
    
    if check_pinch(right_thumb, right_finger_tip):
        midpoint = [int(w * (right_thumb[0][0] + right_finger_tip[0][0]) / 2), 
                    int(h * (right_thumb[0][1] + right_finger_tip[0][1]) / 2)]
        current_path.append(midpoint)
    else:
        if current_path:
            paths.append(current_path)
            current_path = []

    for i in range(len(current_path) - 1):
        cv2.line(black_display, 
                (current_path[i][0], current_path[i][1]), 
                (current_path[i + 1][0], current_path[i + 1][1]), 
                color=(255, 255, 0), thickness=5)

    for path in paths:
        for i in range(len(path) - 1):
            cv2.line(black_display, 
                    (path[i][0], path[i][1]), 
                    (path[i + 1][0], path[i + 1][1]), 
                    color=(255, 255, 0), thickness=5)

    cv2.imshow('MediaPipe Hands', black_display)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
