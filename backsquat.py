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

def vector_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

cap = cv2.VideoCapture(r'C:/Users/NBODT/Desktop/DE/data/Back Squat.mp4')
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

        frame = cv2.resize(frame, (640, 480))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # จุดสำคัญ
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                # --- Knee angles ---
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

                # --- Count Squats ---
                if avg_knee_angle <= 90 and stage == 'up':
                    count += 1
                    stage = 'down'
                elif avg_knee_angle >= 160 and stage == 'down':
                    stage = 'up'

                move_status = 'Standing' if stage == 'up' else 'Squatting'
                if 70 <= avg_knee_angle <= 160:
                    pose_status = "Good squat!"
                    color = (0, 255, 0)
                elif avg_knee_angle < 70:
                    pose_status = "Squat too deep!"
                    color = (0, 0, 255)
                else:
                    pose_status = "Not deep enough!"
                    color = (0, 165, 255)
                # ไม่แสดงสถานะขาตรง/ขาเบี้ยวอีกต่อไป

                # --- Parallelism: ความขนานของเส้นไหล่กับสะโพก ---
                shoulder_vec = np.array(right_shoulder) - np.array(left_shoulder)
                hip_vec = np.array(right_hip) - np.array(left_hip)
                parallel_angle = vector_angle(shoulder_vec, hip_vec)
                if parallel_angle < 10 or abs(parallel_angle - 180) < 10:
                    parallel_status = "Parallel"
                    color_parallel = (0, 255, 0)
                else:
                    parallel_status = "Not parallel"
                    color_parallel = (0, 0, 255)

                # --------- Display ---------
                cv2.putText(image, f'Left Knee Angle: {int(left_knee_angle)} deg',
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Right Knee Angle: {int(right_knee_angle)} deg',
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Parallel Angle: {parallel_angle:.2f} deg',
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color_parallel, 2, cv2.LINE_AA)
                cv2.putText(image, f'Shoulder-Hip: {parallel_status}',
                            (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color_parallel, 2, cv2.LINE_AA)
                cv2.putText(image, f'Count: {count}',
                            (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, move_status,
                            (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 128, 0), 2, cv2.LINE_AA)
                cv2.putText(image, pose_status,
                            (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2, cv2.LINE_AA)
                # ไม่แสดงข้อความ Legs straight/Legs bent

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
