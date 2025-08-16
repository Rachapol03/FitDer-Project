import cv2
import mediapipe as mp
import numpy as np

# คำนวณมุมข้อศอก
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# ประเมินท่า pushdown โดยดูแค่ลักษณะของหลัง (ไหล่-สะโพก-เข่า)
def classify_pushdown(shoulder, hip, knee):
    back_angle = calculate_angle(shoulder, hip, knee)
    if back_angle > 160:
        pose_quality = "Correct pose"
    else:
        pose_quality = "Not correct"
    # stage ยังใช้มุมข้อศอก
    return pose_quality, back_angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ใส่ 0 ถ้าต้องการใช้กล้อง
cap = cv2.VideoCapture('C:\งาน\HOW TO_ Cable Triceps Pushdown _ 3 Golden Rules (FOR GROWTH) (online-video-cutter.com).mp4')

pushdown_counter = 0
pushdown_stage = "Up"
counted = False

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
            shoulder = [landmarks[12].x, landmarks[12].y]  # right_shoulder
            elbow = [landmarks[14].x, landmarks[14].y]      # right_elbow
            wrist = [landmarks[16].x, landmarks[16].y]      # right_wrist
            hip = [landmarks[24].x, landmarks[24].y]        # right_hip
            knee = [landmarks[26].x, landmarks[26].y]       # right_knee

            # วัดมุมศอกแนบตัว (ระหว่างสะโพก-ศอก-ข้อมือ)
            elbow_hip_angle = calculate_angle(hip, elbow, wrist)
            # วัดมุมลำตัว (ไหล่-สะโพก-เข่า)
            torso_angle = calculate_angle(shoulder, hip, knee)

            # stage สำหรับ pushdown
            if elbow_hip_angle < 70:
                stage = "Down"
            elif elbow_hip_angle > 150:
                stage = "Up"
            else:
                stage = "Middle"

            # ประเมิน correct pose จากหลัง
            pose_quality, back_angle = classify_pushdown(shoulder, hip, knee)

            if stage == "Down" and pushdown_stage == "Up":
                pushdown_stage = "Down"
                counted = False

            elif stage == "Up" and pushdown_stage == "Down" and not counted:
                if elbow_hip_angle > 150:
                    pushdown_counter += 1
                    counted = True
                    pushdown_stage = "Up"

            color = (0, 255, 0) if pose_quality == "Correct pose" else (0, 0, 255)

            # เช็คองศาแขนถ้าเกินลำตัว
            # เช็คว่าแขนถูกต้องถ้าองศาอยู่ระหว่าง 25 ถึง 180
            if 0 <= elbow_hip_angle <= 180:
                arm_status = "Arm angle correct"
                arm_color = (0, 255, 0)
            else:
                arm_status = "Arm angle incorrect"
                arm_color = (0, 0, 255)

            cv2.putText(image, f"Tricep Pushdown Count: {pushdown_counter}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(image, f"Elbow-Hip Angle: {int(elbow_hip_angle)} deg", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Stage: {stage}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)
            cv2.putText(image, f"{pose_quality} (Back Angle: {int(back_angle)} deg)", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(image, arm_status + f" (diff: {int(abs(elbow_hip_angle-torso_angle))} deg)",
                        (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, arm_color, 2)

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
        cv2.imshow('Tricep Pushdown Counter', image)

        key = cv2.waitKey(10) & 0xFF
        if key in [ord('q'), ord('Q')]:
            break

cap.release()
cv2.destroyAllWindows()
