import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def classify_pushup(shoulder, elbow, wrist, hip):
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    body_slope = abs(shoulder[1] - hip[1])
    # Middle: ระหว่าง Up กับ Down
    if elbow_angle < 100 and body_slope < 0.1:
        return "Down", "Good", elbow_angle
    elif 100 <= elbow_angle < 160:
        return "Middle", "Bad", elbow_angle
    else:
        return "Up", "Bad", elbow_angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# เปลี่ยนเป็นวิดีโอของ push-up ได้เลย
cap = cv2.VideoCapture("How to do a Push-Up _ Proper Form & Technique _ NASM.mp4")

pushup_counter = 0
pushup_stage = None

with mp_pose.Pose(min_detection_confidence=0.5) as pose:
    try:
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
                shoulder = [landmarks[12].x, landmarks[12].y]
                elbow = [landmarks[14].x, landmarks[14].y]
                wrist = [landmarks[16].x, landmarks[16].y]
                hip = [landmarks[24].x, landmarks[24].y]

                stage, form_quality, angle = classify_pushup(shoulder, elbow, wrist, hip)

                # เปลี่ยนการนับจากตอนลงเป็นตอนขึ้น
                if stage == 'Up' and pushup_stage == 'Down':
                    pushup_counter += 1
                    pushup_stage = 'Up'
                elif stage == 'Down':
                    pushup_stage = 'Down'

                color = (0, 255, 0) if form_quality == "Good" else (0, 0, 255)

                cv2.putText(image, f"Push-up Count: {pushup_counter}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                cv2.putText(image, f"Elbow Angle: {int(angle)} deg ({form_quality})", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(image, f"Stage: {stage}", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)

            except:
                pass

            cv2.putText(image, "Press Q to Quit", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            cv2.imshow('Push-up Counter', image)

            key = cv2.waitKey(10) & 0xFF
            if key in [ord('q'), ord('Q')]:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()