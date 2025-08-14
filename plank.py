import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def assess_plank(shoulder, hip, ankle):
    angle = calculate_angle(shoulder, hip, ankle)
    if 160 <= angle <= 180:
        return "Good", angle
    else:
        return "Bad", angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("Core Exercise_ Plank (online-video-cutter.com).mp4")  # หรือใช้กล้องเปลี่ยนเป็น cap = cv2.VideoCapture(0)

start_time = None
good_form_time = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[24].x, landmarks[24].y]  # right shoulder
            hip = [landmarks[26].x, landmarks[26].y]       # right hip
            ankle = [landmarks[28].x, landmarks[28].y]     # right ankle

            form, angle = assess_plank(shoulder, hip, ankle)

            # จับเวลาถ้าท่าถูกต้อง
            if form == "Good":
                if start_time is None:
                    start_time = time.time()
                else:
                    good_form_time = time.time() - start_time
            else:
                start_time = None
                good_form_time = 0

            color = (0, 255, 0) if form == "Good" else (0, 0, 255)

            cv2.putText(image, f"Plank Form: {form}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(image, f"Hip Angle: {int(angle)} deg", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, f"Hold Time: {int(good_form_time)} sec", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # --- วาด keypoint และ line ---
            h, w, _ = image.shape
            # วาดเส้นระหว่าง keypoint ตามโครงสร้างร่างกาย
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # วาดจุดและเลขกำกับ
            for idx, lm in enumerate(landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 6, (0, 255, 255), -1)
                cv2.putText(image, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        except:
            pass

        cv2.putText(image, "Press Q to Quit", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        cv2.imshow('Plank Form Checker', image)

        key = cv2.waitKey(10) & 0xFF
        if key in [ord('q'), ord('Q')]:
            break

cap.release()
cv2.destroyAllWindows()
