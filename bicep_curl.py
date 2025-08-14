import cv2
import mediapipe as mp
import math
import numpy as np

def calculate_angle(a, b, c):
    a = [a[0], a[1]]
    b = [b[0], b[1]]
    c = [c[0], c[1]]
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_standing_straight(landmarks, mp_pose, threshold=0.08):
    """
    ตรวจสอบว่าแนวไหล่ซ้าย-สะโพกซ้ายขนานกับแนวตั้ง (แกน y) หรือไม่
    threshold: ค่าความเอียงสูงสุดที่ยอมรับ (unit: normalized x)
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    # ถ้า x ของไหล่กับสะโพกต่างกันมาก แปลว่าเอียง
    if abs(left_shoulder.x - left_hip.x) > threshold:
        return False
    return True

def vector_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def is_upperarm_parallel_to_torso(landmarks, mp_pose, angle_threshold=40):
    """
    ตรวจสอบว่าเวกเตอร์ไหล่->ศอก ขนานกับเวกเตอร์ไหล่->สะโพก (แนวลำตัว) หรือไม่
    angle_threshold: องศาสูงสุดที่ยอมรับว่า "ขนาน" (เช่น 20 องศา)
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    # เวกเตอร์ไหล่->ศอก
    v1 = [left_elbow.x - left_shoulder.x, left_elbow.y - left_shoulder.y]
    # เวกเตอร์ไหล่->สะโพก
    v2 = [left_hip.x - left_shoulder.x, left_hip.y - left_shoulder.y]
    # มุมระหว่างเวกเตอร์
    angle = vector_angle(v1, v2)
    # ขนาน = มุมใกล้ 0 หรือ 180
    if angle < angle_threshold or abs(angle-180) < angle_threshold:
        return True
    return False

cap = cv2.VideoCapture('biceb_curl_right.mp4')
mp_pose = mp.solutions.pose

with mp_pose.Pose(static_image_mode=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    count = 0
    stage = 'up'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ปรับขนาดสำหรับวิดีโอแนวตั้ง (portrait)
        frame = cv2.resize(frame, (480, 640))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # จุดสำคัญสำหรับ double bicep curl
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # --- Elbow angles ---
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

                # --- Count Bicep Curls ---
                up_threshold = 80
                down_threshold = 140
                move_status = 'Curl Down' if stage == 'down' else 'Curl Up'
                standing_ok = is_standing_straight(landmarks, mp_pose)
                upperarm_parallel = is_upperarm_parallel_to_torso(landmarks, mp_pose)
                parallel_color = (0,255,0) if upperarm_parallel else (0,0,255)
                parallel_msg = 'Arm OK' if upperarm_parallel else 'Arm not parallel!'
                standing_color = (0,255,0) if standing_ok else (0,0,255)
                standing_msg = 'Standing OK' if standing_ok else 'Not straight!'
                if left_elbow_angle >= down_threshold and stage == 'up':
                    stage = 'down'
                    move_status = 'Curl Down'
                elif left_elbow_angle <= up_threshold and stage == 'down':
                    if standing_ok and upperarm_parallel:
                        count += 1
                        move_status = 'Curl Up'
                    elif not standing_ok:
                        move_status = 'Bad Stand'
                    elif not upperarm_parallel:
                        move_status = 'Bad Arm'
                    stage = 'up'

                move_status = 'Curl Up' if stage == 'up' else 'Curl Down'
                if 20 <= avg_elbow_angle <= 160:
                    pose_status = "Good curl!"
                    color = (0, 255, 0)
                elif avg_elbow_angle < 50:
                    pose_status = "Curl too much!"
                    color = (0, 0, 255)
                else:
                    pose_status = "Not enough curl!"
                    color = (0, 165, 255)

                # --------- Display ---------
                cv2.putText(image, f'Left Elbow Angle: {int(left_elbow_angle)} deg',
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Right Elbow Angle: {int(right_elbow_angle)} deg',
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Count: {count}',
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, move_status,
                            (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 128, 0), 2, cv2.LINE_AA)
                cv2.putText(image, standing_msg,
                            (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            standing_color, 2, cv2.LINE_AA)
                cv2.putText(image, parallel_msg,
                            (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            parallel_color, 2, cv2.LINE_AA)
                cv2.putText(image, pose_status,
                            (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2, cv2.LINE_AA)

            except Exception as e:
                pass

            # วาด keypoints เฉพาะส่วนที่ไม่ใช่ใบหน้า
            face_landmarks = set(list(range(0,11)) + list(range(15,19)))
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx in face_landmarks or end_idx in face_landmarks:
                    continue
                start = results.pose_landmarks.landmark[start_idx]
                end = results.pose_landmarks.landmark[end_idx]
                h, w, _ = image.shape
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(image, (x1, y1), (x2, y2), (0,255,255), 2)
            for idx, lm in enumerate(landmarks):
                if idx in face_landmarks:
                    continue
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (0,255,255), -1)
                cv2.putText(image, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Pose Keypoints', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
