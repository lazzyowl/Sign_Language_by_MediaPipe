import pickle
import cv2
import mediapipe as mp
import numpy as np

# 1. Load the model
# Make sure this filename matches what you saved! (e.g., 'model.p' or 'my_asl_model.p')
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: 'model.p' not found. Did you name it something else in train_classifier.py?")
    exit()

# 2. Setup Camera and MediaPipe
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# 3. Define the Dictionary
# This maps the "Folder Name" -> "Display Text"
# Using strings ('0') allows us to mix numbers and letters without crashing.
labels_dict = {
    # Numbers
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
    'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
    'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O',
    'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
    'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
    'HELLO':'HELLO','GOODBYE':'GOODBYE','I LOVE YOU':'I LOVE YOU',
    'PLEASE':'PLEASE','NO':'NO','YES':'YES','SORRY':'SORRY',
    'THANK YOU':'THANK YOU',

   
}

print("System Ready. Press 'Q' to quit.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # ---------------------------------------------------------
        # CRITICAL FIX: Only process the FIRST hand detected [0]
        # This prevents the "inhomogeneous shape" crash.
        # ---------------------------------------------------------
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Collect coordinates
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        # Normalize coordinates (x - min)
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        # 4. Prediction Logic
        # We assume data_aux is correct shape (42,)
        prediction = model.predict([np.asarray(data_aux)])
        
        # Convert result to string immediately to handle both 0 and 'A'
        predicted_label = str(prediction[0])

        # 5. Safe Dictionary Lookup
        try:
            # Look up the label in our dictionary
            predicted_character = labels_dict[predicted_label]
        except KeyError:
            # If the model predicts a label we forgot to add to the dictionary,
            # just display the raw label itself so the code doesn't crash.
            predicted_character = predicted_label

        # Draw UI
        cv2.rectangle(frame, (0, 0), (100, 100), (0, 0, 0), -1)
        cv2.putText(frame, predicted_character, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    
    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()