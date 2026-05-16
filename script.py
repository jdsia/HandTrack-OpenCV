import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque, Counter

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# All the bone connections between landmark points
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17)              # palm knuckle line
]

def draw_hand_skeleton(frame, hand_landmarks):
    h, w, _ = frame.shape

    # Convert normalised coords (0.0–1.0) to pixel positions
    points = [
        (int(lm.x * w), int(lm.y * h))
        for lm in hand_landmarks
    ]

    # Draw lines between connected joints
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

    # Draw a circle on each joint
    for i, (x, y) in enumerate(points):
        # Fingertips (4, 8, 12, 16, 20) get a bigger coloured dot
        if i in (4, 8, 12, 16, 20):
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (x, y), 6, (0, 180, 255), -1)

def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]
    pip  = [6, 10, 14, 18]

    count = sum(
        hand_landmarks[t].y < hand_landmarks[p].y
        for t, p in zip(tips, pip)
    )

    wrist_x = hand_landmarks[0].x
    index_knuckle_x = hand_landmarks[5].x

    if index_knuckle_x > wrist_x:
        if hand_landmarks[4].x > hand_landmarks[3].x:
            count += 1
    else:
        if hand_landmarks[4].x < hand_landmarks[3].x:
            count += 1

    return count

history = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        n = count_fingers(results.hand_landmarks[0])
        history.append(n)
        draw_hand_skeleton(frame, results.hand_landmarks[0])  # draw skeleton

    if history:
        stable_n = Counter(history).most_common(1)[0][0]
        cv2.putText(frame, f'Fingers up: {stable_n}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()