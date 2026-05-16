import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque, Counter

# --- Setup ---
# Download the model from:
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# and place it in the same folder as this script.

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# Open connection to default webcam
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    # Tips of each finger (index, middle, ring, pinky)
    tips = [8, 12, 16, 20]
    # Knuckles
    pip = [6, 10, 14, 18]

    count = sum(
        hand_landmarks[t].y < hand_landmarks[p].y
        for t, p in zip(tips, pip)
        # zip returns an iterator of tuples (tip, pip)
        # rows into cols and cols into rows.
    )

    # Thumb uses x-axis instead of y-axis
    if hand_landmarks[4].x < hand_landmarks[3].x:
        count += 1

    return count  # Return number of fingers up

history = deque(maxlen=10)  # Store last 10 counts

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # flips cam
    #if not ret:
    #    break
    # MediaPipe expects RGB, OpenCV uses BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Wrap frame in a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        # Grab landmarks for the first detected hand
        n = count_fingers(results.hand_landmarks[0])
        history.append(n)  # Add current count to history

        # Draw the finger count as text on the frame
    if history:
      stable_n = Counter(history).most_common(1)[0][0]  # Get most common count
      cv2.putText(frame, f'Fingers up: {n}', (50, 50),
          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  

    # Display the frame in a window called 'Hand'
    cv2.imshow('Hand', frame)

    # If user presses 'q', quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()