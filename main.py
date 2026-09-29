import cv2
import math
import time
from pathlib import Path

import mediapipe as mp
import pyautogui

import mouse_config

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "hand_landmarker.task")

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

screen_w, screen_h = pyautogui.size()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

prev_x, prev_y = 0, 0
prev_scroll_y = 0
is_left_clicked = False
is_right_clicked = False
is_dragging = False
pinch_start_time = 0

def get_pixel_coords(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

def calculate_distance_pt(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def display_frame(frame, window_name="AI Virtual Mouse Control"):
    try:
        cv2.imshow(window_name, frame)
        return True
    except cv2.error:
        return False


def read_keypress():
    try:
        return cv2.waitKey(1) & 0xFF
    except cv2.error:
        return None


def main():
    prev_x = prev_y = 0
    prev_scroll_y = 0
    is_left_clicked = False
    is_right_clicked = False
    is_dragging = False
    pinch_start_time = 0

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )

    cap = cv2.VideoCapture(0)
    gui_available = True
    print("[INFO] Initializing Precision Virtual Mouse Engine...")

    if not cap.isOpened():
        print("[ERROR] Unable to access the webcam.")
        return

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                margin = mouse_config.FRAME_MARGIN

                cv2.rectangle(frame, (margin, margin), (w - margin, h - margin), (255, 255, 0), 2)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = landmarker.detect(mp_image)

                if detection_result.hand_landmarks:
                    hand_landmarks = detection_result.hand_landmarks[0]

                    index_pt = get_pixel_coords(hand_landmarks[mouse_config.INDEX_FINGER_TIP], w, h)
                    thumb_pt = get_pixel_coords(hand_landmarks[mouse_config.THUMB_TIP], w, h)
                    middle_pt = get_pixel_coords(hand_landmarks[mouse_config.MIDDLE_FINGER_TIP], w, h)
                    wrist_pt = get_pixel_coords(hand_landmarks[mouse_config.WRIST], w, h)
                    middle_mcp_pt = get_pixel_coords(hand_landmarks[mouse_config.MIDDLE_FINGER_MCP], w, h)

                    cv2.circle(frame, index_pt, 7, (255, 0, 0), -1)
                    cv2.circle(frame, thumb_pt, 7, (0, 255, 0), -1)
                    cv2.circle(frame, middle_pt, 7, (0, 0, 255), -1)

                    hand_scale = calculate_distance_pt(wrist_pt, middle_mcp_pt)
                    if hand_scale < 1.0:
                        hand_scale = 1.0

                    left_pinch_ratio = calculate_distance_pt(index_pt, thumb_pt) / hand_scale
                    right_pinch_ratio = calculate_distance_pt(middle_pt, thumb_pt) / hand_scale
                    scroll_ratio = calculate_distance_pt(index_pt, middle_pt) / hand_scale

                    clamped_x = max(margin, min(index_pt[0], w - margin))
                    clamped_y = max(margin, min(index_pt[1], h - margin))

                    target_x = int(((clamped_x - margin) / (w - 2 * margin)) * screen_w)
                    target_y = int(((clamped_y - margin) / (h - 2 * margin)) * screen_h)

                    alpha = mouse_config.SMOOTHING_FACTOR
                    curr_x = int(alpha * target_x + (1 - alpha) * prev_x)
                    curr_y = int(alpha * target_y + (1 - alpha) * prev_y)

                    if abs(curr_x - prev_x) < mouse_config.DEADZONE_PIXELS:
                        curr_x = prev_x
                    if abs(curr_y - prev_y) < mouse_config.DEADZONE_PIXELS:
                        curr_y = prev_y

                    if scroll_ratio < mouse_config.NORM_SCROLL_RATIO and left_pinch_ratio > mouse_config.NORM_CLICK_RATIO:
                        cv2.line(frame, index_pt, middle_pt, (255, 0, 255), 3)
                        cv2.putText(frame, "SCROLL MODE", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

                        if prev_scroll_y != 0:
                            delta_y = prev_scroll_y - index_pt[1]
                            if abs(delta_y) > 4:
                                pyautogui.scroll(int(delta_y * 3))
                        prev_scroll_y = index_pt[1]

                    else:
                        prev_scroll_y = 0

                        if is_dragging:
                            pyautogui.dragTo(curr_x, curr_y, button='left')
                        else:
                            pyautogui.moveTo(curr_x, curr_y)

                        prev_x, prev_y = curr_x, curr_y

                        if left_pinch_ratio < mouse_config.NORM_CLICK_RATIO:
                            if not is_left_clicked:
                                is_left_clicked = True
                                pinch_start_time = time.time()
                            elif time.time() - pinch_start_time > mouse_config.DRAG_DELAY:
                                if not is_dragging:
                                    pyautogui.mouseDown(button='left')
                                    is_dragging = True
                                cv2.line(frame, index_pt, thumb_pt, (0, 255, 255), 3)
                                cv2.putText(frame, "DRAGGING", (30, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            else:
                                cv2.line(frame, index_pt, thumb_pt, (0, 255, 0), 3)
                        else:
                            if is_left_clicked:
                                if is_dragging:
                                    pyautogui.mouseUp(button='left')
                                    is_dragging = False
                                else:
                                    pyautogui.click()
                                    cv2.putText(frame, "LEFT CLICK", (30, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                is_left_clicked = False

                        if right_pinch_ratio < mouse_config.NORM_RIGHT_CLICK_RATIO:
                            cv2.line(frame, middle_pt, thumb_pt, (0, 0, 255), 3)
                            if not is_right_clicked:
                                pyautogui.rightClick()
                                is_right_clicked = True
                                cv2.putText(frame, "RIGHT CLICK", (30, 90),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        else:
                            is_right_clicked = False

                else:
                    cv2.putText(frame, "NO HAND DETECTED", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if gui_available:
                    if not display_frame(frame):
                        gui_available = False
                        print("[INFO] GUI display backend unavailable; continuing in headless mode.")
                    else:
                        key = read_keypress()
                        if key == ord('q'):
                            break
                else:
                    time.sleep(0.033)
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("[INFO] Engine shutdown cleanly.")


if __name__ == "__main__":
    main()