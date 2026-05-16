import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter, deque
import math
from pynput.mouse import Controller, Button
import time

mouse = Controller()
screen_w, screen_h = 1920, 1200
# state for mouse
left_button_down = False
last_click_time = 0
cooldown = 0.5 #secs

# for smoothing
prev_x, prev_y = 0, 0
alpha = 0.2

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


# helper function for mapping for scroll
def map_range(val, in_min, in_max):
    return (val - in_min) / (in_max - in_min)


def move_cursor(hand_landmarks):
    global prev_x, prev_y

    index = hand_landmarks[8]

    # 1. RAW normalized coords from MediaPipe (0–1)
    x = index.x
    y = index.y

    # 2. Define usable control region (prevents edge distortion)
    x_min, x_max = 0.2, 0.8
    y_min, y_max = 0.2, 0.8

    # 3. Clamp into control region
    x = max(x_min, min(x_max, x))
    y = max(y_min, min(y_max, y))

    # 4. Normalize within region → 0–1
    norm_x = (x - x_min) / (x_max - x_min)
    norm_y = (y - y_min) / (y_max - y_min)

    # 5. Map to screen space
    target_x = norm_x * screen_w
    target_y = norm_y * screen_h

    # 6. Smooth (EMA filter)
    smooth_x = prev_x + (target_x - prev_x) * alpha
    smooth_y = prev_y + (target_y - prev_y) * alpha

    # 7. Apply to mouse
    mouse.position = (int(smooth_x), int(smooth_y))

    # 8. Update state
    prev_x, prev_y = smooth_x, smooth_y

def handle_left_click_hold(is_pinching):
    global left_button_down

    if is_pinching and not left_button_down:
        mouse.press(Button.left)
        left_button_down = True

    elif not is_pinching and left_button_down:
        mouse.release(Button.left)
        left_button_down = False

def safe_click(button):
    global last_click_time

    now = time.time()
    if now - last_click_time > cooldown:
        mouse.click(button)
        last_click_time = now

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

def detect_pinch(hand_landmarks):
    thumb = hand_landmarks[4]
    index = hand_landmarks[8]

    distance = math.sqrt(
        (thumb.x - index.x)**2 +
        (thumb.y - index.y)**2
    )
    return distance < 0.05


# Added detect_spread function to detect hand spread
def detect_spread(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    return distance > 0.2  # Threshold for spread detection


# FIX 1: function was defined as three_fingers_up but called as is_three_fingers_up
def three_fingers_up(hand_landmarks):
    tips = [8, 12, 16]
    pip = [6, 10, 14]
    return sum(
        hand_landmarks[t].y < hand_landmarks[p].y
        for t, p in zip(tips, pip)
    ) == 3

def handle_scroll(hand_landmarks):
    pass  # Removed scrolling logic


history = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        hand = results.hand_landmarks[0]

        three_fingers = three_fingers_up(hand)  # FIX 1: corrected function name

        move_cursor(hand)

        n = count_fingers(hand)
        history.append(n)

        draw_hand_skeleton(frame, hand)

        pinching = detect_pinch(hand)
        if pinching:
            cv2.putText(
                frame,
                "PINCHED FINGERS",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        handle_left_click_hold(pinching)

        stable_n = Counter(history).most_common(1)[0][0]
        if detect_spread(hand) and stable_n == 2:
            cv2.putText(
                frame,
                "SPREAD FINGERS",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            safe_click(Button.right)

    if history:
        stable_n = Counter(history).most_common(1)[0][0]
        cv2.putText(frame, f'Fingers up: {stable_n}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()