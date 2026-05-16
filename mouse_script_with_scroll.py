import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque, Counter
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

# scrolling vars
scrolling = False
scrollhistory = deque(maxlen=10)

# swipe detection
swipe_buffer = deque(maxlen=8)  # recent Y positions of index finger
swipe_threshold = 0.04          # min Y change to count as a swipe (tune this)
swipe_cooldown = 0.3            # secs between swipes (prevents double-firing)
last_swipe_time = 0
scroll_amount = 5               # how many units to scroll per swipe (tune this)

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

# Add scrolling functionality
def handle_scroll(hand_landmarks):
    global last_swipe_time

    index_y = hand_landmarks[8].y
    swipe_buffer.append(index_y)

    if len(swipe_buffer) < 2:
        return

    delta_y = swipe_buffer[-1] - swipe_buffer[0]

    if abs(delta_y) > swipe_threshold:
        current_time = time.time()
        if current_time - last_swipe_time > swipe_cooldown:
            direction = -1 if delta_y > 0 else 1
            mouse.scroll(0, direction * scroll_amount)
            last_swipe_time = current_time

# Added detect_spread function to detect hand spread
def detect_spread(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    return distance > 0.2  # Threshold for spread detection