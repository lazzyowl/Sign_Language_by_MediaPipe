import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    # Skip non-directory files
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            # Iterate through ALL detected hands (up to 2)
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

            # Normalize data based on the minimum value found in the set
            for i in range(len(x_)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

            # --- CRITICAL FIX: PADDING ---
            # The model expects a fixed size (84 features for 2 hands).
            # If we only found 1 hand (42 features), we add 42 zeros.
            expected_features = 84 # 21 landmarks * 2 coords * 2 hands
            
            while len(data_aux) < expected_features:
                data_aux.append(0.0)

            # If accidental noise makes it larger, truncate it
            data_aux = data_aux[:expected_features]

            data.append(data_aux)
            labels.append(dir_)

# Save to pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
print("Dataset created successfully! (Compatible with 1 or 2 hands)")