import cv2
import mediapipe as mp
import math
import os

def calculate_angle(a, b, c):
    a = [a[0], a[1]]
    b = [b[0], b[1]]
    c = [c[0], c[1]]

    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(static_image_mode=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    count = 0
    stage = 'up'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            try:

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # มุมเดิม (left_knee, left_hip, ankle)
                hip23_angle = calculate_angle(left_knee, left_hip, ankle)
                leg_angle = calculate_angle(left_hip, left_knee, ankle)

                # มุม hip23 ใหม่: LEFT_HIP - RIGHT_HIP - LEFT_KNEE
                hip23_new_angle = calculate_angle(right_hip, left_hip, left_knee)


                # --- นับจำนวนครั้งตามลูปขึ้น-ลง ---
                if hip23_new_angle <= 90 and stage == 'up':
                    count += 1
                    stage = 'down'
                elif hip23_new_angle >= 120 and stage == 'down':
                    stage = 'up'


                # สถานะ (up/down)
                move_status = 'Leg Up' if stage == 'up' else 'Leg Down'

                # Status ท่าทาง (ใช้ hip23_new_angle)
                if 70 <= hip23_new_angle <= 130:
                    pose_status = "Correct pose!"
                    color = (0, 255, 0)
                elif hip23_new_angle > 130:
                    pose_status = "Leg too wide!"
                    color = (0, 165, 255)
                else:
                    pose_status = "Leg not high enough!"
                    color = (0, 0, 255)

                # Status ขาตรง/ขาเบี้ยว
                if 170 <= leg_angle <= 190:
                    leg_status = "Leg straight"
                    leg_status_color = (0, 255, 0)
                else:
                    leg_status = "Leg not straight!"
                    leg_status_color = (0, 0, 255)

                # --------- แสดงผล ---------

                cv2.putText(image, f'Leg Angle: {int(leg_angle)} deg',
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Hip23 Angle (LH-RH-LK): {int(hip23_new_angle)} deg',
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Count: {count}',
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, move_status,
                            (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 128, 0), 2, cv2.LINE_AA)
                cv2.putText(image, pose_status,
                            (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2, cv2.LINE_AA)
                cv2.putText(image, leg_status,
                            (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            leg_status_color, 2, cv2.LINE_AA)

            except Exception as e:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Optional: draw landmark indices for debugging
            h, w, _ = image.shape
            for idx, lm in enumerate(landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(image, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Pose Keypoints', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
