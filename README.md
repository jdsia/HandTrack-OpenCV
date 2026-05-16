# Hand Tracking with OpenCV & MediaPipe

Real-time hand pose detection and visualization using OpenCV and MediaPipe. This project captures video from your webcam, detects hand landmarks, and displays the hand skeleton overlay.

## Prerequisites

- Python 3.8+
- Webcam/camera device
- `hand_landmarker.task` model file (MediaPipe task model)

## Installation

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 3. Upgrade pip and Install Dependencies

```bash
python -m pip install --upgrade pip
pip install mediapipe opencv-python pygame
```

## Running the Project

With the virtual environment activated:

```bash
python script.py
```

The script will:
- Open your webcam
- Detect your hand in real-time
- Display hand landmarks (joints) as circles
- Connect joints with lines to show the hand skeleton
- Fingertips are highlighted with larger green circles

Press `q` to exit the application.

## Project Structure

```
.
├── script.py                 # Main hand tracking script
├── hand_landmarker.task      # MediaPipe hand detection model
└── README.md                 # This file
```

## How It Works

1. **Capture**: Reads video frames from your webcam
2. **Detect**: Uses MediaPipe HandLandmarker to find 21 hand landmark points
3. **Visualize**: Draws connections between landmarks to form a hand skeleton
4. **Display**: Shows the result in real-time

## Key Features

- Real-time hand detection from webcam feed
- 21-point hand pose landmarks per detected hand
- Skeletal visualization with bone connections
- Finger-specific highlighting (fingertips in green)

## Troubleshooting

### "hand_landmarker.task not found"
Make sure the `hand_landmarker.task` model file is in the same directory as `script.py`. Download it from [MediaPipe's official task models](https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker).

