import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
cap =cv2.VideoCapture(r'C:/Users/NBODT/Desktop/DE/data/ALT RLUNG.mp4')


count = 0
direction = 0  # 0 = down, 1 = up
good_form = True
last_leg = None  # 'left' or 'right'

def calculate_angle(a, b, c):
    a = np.array(a)  # first
    b = np.array(b)  # mid
    c = np.array(c)  # end
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (479, 480))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        # มุมเข่าซ้าย
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        left_angle = calculate_angle(l_hip, l_knee, l_ankle)

        # มุมเข่าขวา
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        right_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # ตรวจจับขาที่กำลังถอยหลัง (ข้างที่งอมากกว่า)
        if left_angle < right_angle:
            active_leg = 'left'
            active_angle = left_angle
        else:
            active_leg = 'right'
            active_angle = right_angle

        # เงื่อนไขการนับ: เข่างอ < 90 และสลับข้าง
        if active_angle < 90:
            if direction == 0 and last_leg != active_leg:
                count += 1
                direction = 1
                last_leg = active_leg
        if active_angle > 160:
            direction = 0

        # ตรวจสอบฟอร์ม (เข่าไม่ต่ำกว่า 60 และไม่เหยียดเกิน 170)
        if active_angle < 60 or active_angle > 170:
            good_form = False
        else:
            good_form = True

        # (ย้ายการแสดงองศาไปไว้ใต้ Count ด้านบนซ้าย)

        # แสดงผล
        y_base = 40
        y_step = 35
        cv2.putText(image, f'Count: {count}', (50, y_base), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(image, f'R: {int(right_angle)}', (50, y_base + y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        cv2.putText(image, f'L: {int(left_angle)}', (50, y_base + 2*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)


        # ตรวจสอบว่าเข่าไหนอยู่ข้างหน้า/ข้างหลัง (y น้อยกว่าคืออยู่ข้างหน้า)
        rknee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        lknee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        if rknee_y < lknee_y:
            knee_status = 'Right knee front'
        elif lknee_y < rknee_y:
            knee_status = 'Left knee front'
        else:
            knee_status = 'Knees aligned'
        cv2.putText(image, knee_status, (50, y_base + 3*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


        # แสดงสถานะยืน/ลง
        status_text = 'Standing' if direction == 0 else 'Down'
        cv2.putText(image, f'Status: {status_text}', (50, y_base + 4*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

        # เงื่อนไขการลง (DOWN) แสดงผลมุมเข่าข้างหน้า
        if direction == 1:
            # เข่าข้างหน้า
            if rknee_y < lknee_y:
                front_angle = right_angle
                back_angle = left_angle
                front_label = 'Right knee'
                back_label = 'Left knee'
            else:
                front_angle = left_angle
                back_angle = right_angle
                front_label = 'Left knee'
                back_label = 'Right knee'
            # เงื่อนไขเข่าข้างหน้า
            if 70 <= front_angle <= 100:
                front_text = 'The knee angle is good'
                front_color = (0,255,0)
            elif front_angle < 70:
                front_text = 'You went too deep'
                front_color = (0,0,255)
            else:
                front_text = 'The knee is in a shallow'
                front_color = (0,165,255)
            cv2.putText(image, front_text, (50, y_base + 5*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, front_color, 2)

            # เงื่อนไขเข่าข้างหลัง
            if 70 <= back_angle <= 110:
                back_text = 'The back knee angle is good'
                back_color = (0,255,0)
            elif back_angle < 70:
                back_text = 'Your back knee went too deep'
                back_color = (0,0,255)
            else:
                back_text = 'The back knee is in a shallow'
                back_color = (0,165,255)
            cv2.putText(image, back_text, (50, y_base + 6*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, back_color, 2)

        # เงื่อนไขการยืน (standing)
        if direction == 0:
            if 170 <= right_angle <= 190 and 170 <= left_angle <= 190:
                stand_text = 'Good Stand'
                stand_color = (0,255,0)
            else:
                stand_text = 'Bad Stand'
                stand_color = (0,0,255)
            cv2.putText(image, stand_text, (50, y_base + 5*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, stand_color, 2)

        # เงื่อนไขการแสดง Form: Good เฉพาะเมื่อ good standing หรือขณะลงทั้งคู่ good
        form_good = False
        if direction == 0:
            if 170 <= right_angle <= 190 and 170 <= left_angle <= 190:
                form_good = True
        elif direction == 1:
            # ต้องใช้ front_angle, back_angle ที่คำนวณไว้ข้างบน
            if ('front_angle' in locals() and 'back_angle' in locals() and
                70 <= front_angle <= 100 and 70 <= back_angle <= 110):
                form_good = True
        cv2.putText(image, f'Form: {"Good" if form_good else "Bad"}', (50, y_base + 7*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if form_good else (0,0,255), 2)

    cv2.imshow('Exercise Counter', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
