import cv2
import mediapipe as mp
import csv
import os
import math
import time
import json
import numpy as np

GESTURES = {
    "1": ("reverse", 1),
    "2": ("front", 2)

}

CSV_FILE = "gesture_dataset.csv"
JSON_FILE = "gesture_labels.json"
CHANGE_THRESHOLD = 0.005
COOLDOWN_SECONDS = 5

# Save the number-to-gesture mapping in JSON.
with open(JSON_FILE, "w") as f:
    json.dump({key: value[0] for key, value in GESTURES.items()}, f, indent=4)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)

cap = cv2.VideoCapture(0)

# 21 landmarks × 3 coordinates × 2 hands = 126 features.
header = []
for hand in ("right", "left"):
    for i in range(21):
        header.extend([f"{hand}_x{i}", f"{hand}_y{i}", f"{hand}_z{i}"])
header.append("label")

if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(header)

sample_counts = {v[0]: 0 for v in GESTURES.values()}
current_gesture = None
current_label = None
last_saved_vector = None
cooldown_until = 0


def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    translated = [[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in landmarks]

    scale = math.sqrt(sum(v * v for v in translated[9]))
    if scale < 1e-8:
        return [0.0] * 63

    return [value / scale for point in translated for value in point]


def extract_two_hand_features(results, frame):
    right_hand = [0.0] * 63
    left_hand = [0.0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            label = handedness.classification[0].label
            vector = normalize_landmarks(hand_lms.landmark)

            if label == "Right":
                right_hand = vector
            else:
                left_hand = vector

    return right_hand + left_hand


print("Press a number to select a gesture:")
for key, (name, _) in GESTURES.items():
    print(f"{key} = {name}")
print("Wait 5 seconds after selection; ESC exits.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key != 255:
            pressed = chr(key)

            if pressed in GESTURES:
                current_gesture, current_label = GESTURES[pressed]
                last_saved_vector = None
                cooldown_until = time.time() + COOLDOWN_SECONDS
                print(
                    f"Selected {current_gesture}. "
                    f"Adjust for {COOLDOWN_SECONDS} seconds..."
                )

        now = time.time()

        if current_gesture is not None and now >= cooldown_until:
            vector = extract_two_hand_features(results, frame)

            if last_saved_vector is None:
                save_sample = True
            else:
                diff = np.mean(np.abs(np.array(vector) - np.array(last_saved_vector)))
                save_sample = diff > CHANGE_THRESHOLD

            if save_sample:
                with open(CSV_FILE, "a", newline="") as f:
                    csv.writer(f).writerow(vector + [current_label])

                last_saved_vector = vector
                sample_counts[current_gesture] += 1
                print(f"Saved {current_gesture}: " f"{sample_counts[current_gesture]}")

        status = (
            f"Current: {current_gesture}"
            if current_gesture is not None
            else "Press 0-3 to select"
        )
        cv2.putText(
            frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

        if current_gesture is not None and now < cooldown_until:
            remaining = max(0, int(cooldown_until - now) + 1)
            cv2.putText(
                frame,
                f"Adjusting: {remaining}s",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        y = 105
        for name, count in sample_counts.items():
            cv2.putText(
                frame,
                f"{name}: {count}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y += 25

        cv2.imshow("Smart Gesture Data Collector", frame)

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
