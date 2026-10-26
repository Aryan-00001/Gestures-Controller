import cv2
import mediapipe as mp
import joblib
import pandas as pd
import json
import math
import os

MODEL_FILE = "gesture_model.pkl"
SIGNAL_MAP_FILE = "gesture_labels.json"

# Load gesture mapping from JSON, e.g. {"0":"reverse","1":"none","2":"front","3":"speed"}
if not os.path.exists(SIGNAL_MAP_FILE):
    raise FileNotFoundError(f"Missing {SIGNAL_MAP_FILE}")

with open(SIGNAL_MAP_FILE, "r") as f:
    signal_to_name = json.load(f)

signal_to_name = {int(k): v for k, v in signal_to_name.items()}
name_to_signal = {v: k for k, v in signal_to_name.items()}

model = joblib.load(MODEL_FILE)

if len(model.feature_names_in_) != 126:
    raise ValueError(
        f"Model expects {len(model.feature_names_in_)} features, "
        "but this tester uses 126. Retrain using the two-hand dataset."
    )

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
prev_prediction = None


def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    translated = [[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in landmarks]

    scale = math.sqrt(sum(v * v for v in translated[9]))
    if scale < 1e-8:
        return [0.0] * 63

    return [v / scale for point in translated for v in point]


def extract_two_hand_features(results, frame):
    right_hand = [0.0] * 63
    left_hand = [0.0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            label = handedness.classification[0].label
            features = normalize_landmarks(hand_lms.landmark)

            if label == "Right":
                right_hand = features
            else:
                left_hand = features

    return right_hand + left_hand


try:
    print("Loaded gesture signals:")
    for signal, name in sorted(signal_to_name.items()):
        print(f"{signal} -> {name}")

    print("Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            vector = extract_two_hand_features(results, frame)

            input_df = pd.DataFrame(
                [vector],
                columns=model.feature_names_in_,
            )

            prediction = int(model.predict(input_df)[0])

            if prediction not in signal_to_name:
                gesture_name = "unknown"
                output = {"gesture": gesture_name, "signal": None}
            else:
                gesture_name = signal_to_name[prediction]
                output = {
                    "gesture": gesture_name,
                    "signal": prediction,
                }

            if prediction != prev_prediction:
                print(json.dumps(output))
                prev_prediction = prediction
        else:
            gesture_name = "no hand"
            if prev_prediction is not None:
                print(json.dumps({"gesture": "no hand", "signal": None}))
                prev_prediction = None

        cv2.putText(
            frame,
            f"Gesture: {gesture_name}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Live Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
