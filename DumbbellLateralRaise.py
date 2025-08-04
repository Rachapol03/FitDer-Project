import cv2
import mediapipe as mp
import numpy as np

# เตรียม mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ฟังก์ชันคำนวณองศาระหว่างจุด 3 จุด
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Load วิดีโอ
cap = cv2.VideoCapture(r'C:/Users/NBODT/Desktop/DE/data/DumbbellLateralRaise.mp4')

counter = 0
stage = None

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # ดึง landmarks

        try:
            landmarks = results.pose_landmarks.landmark
            # เลือก keypoints สำหรับไหล่ ศอก ข้อมือขวา
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # คำนวณองศาที่ข้อศอกกับหัวไหล่
            angle = calculate_angle(shoulder, elbow, wrist)

            # เงื่อนไข Dumbbell Lateral Raise:
            # 1. ตำแหน่งขึ้น - แขนตั้งฉากกับลำตัว (ประมาณ 70-110 องศา)
            # 2. ตำแหน่งลง - แขนชิดลำตัว (ประมาณ 10-30 องศา)
            feedback = "ผิด"
            if 70 < angle < 110:
                feedback = "ถูกต้อง"
                stage = "up"
            if angle < 30 and stage == "up":
                stage = "down"
                counter += 1
                print(f"นับได้ {counter} ครั้ง")
            # แสดง feedback บนวิดีโอ
            cv2.putText(frame, f'Angle: {int(angle)}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Feedback: {feedback}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if feedback=="ถูกต้อง" else (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Count: {counter}', (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            # วาดเส้นและจุด mediapipe pose ลงบน frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        except Exception as e:
            pass

        # (Optional) โชว์ผลบนวิดีโอ
        cv2.imshow('Mediapipe Feed', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print(f"จำนวนครั้งที่ทำ Dumbbell Lateral Raise: {counter}")
