# 🖐️ AI Virtual Mouse with Dynamic Hand Normalization

An adaptive, computer-vision-based touchless mouse interface built using Python, OpenCV, and MediaPipe. Features real-time landmark tracking, exponential smoothing, and hand-scale dynamic normalization to maintain click accuracy across varying camera distances.

---

## 🌟 Key Features

* **Cursor Movement:** Tracks index finger tip coordinates with exponential moving average smoothing and deadzone filtering to reduce micro-tremors.
* **Dynamic Pinch Normalization:** Automatically scales pinch distance thresholds against the user's physical hand size (Wrist to Middle MCP), maintaining precision regardless of distance from the webcam.
* **Left Click & Drag:** Quick index-thumb pinch for single click; holding the pinch (>0.35s) latches into drag mode for selecting text or moving windows.
* **Right Click:** Quick middle-thumb pinch triggers standard right-click actions.
* **Scroll Mode:** Bringing index and middle fingers together enters vertical scrolling mode.

---

## 🛠️ Prerequisites & Setup

### 1. Install Dependencies
```bash
pip install opencv-python mediapipe pyautogui