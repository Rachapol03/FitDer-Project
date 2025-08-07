import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(r'DumbbellLateralRaise.mp4')
#cap = cv2.VideoCapture(0)
count = 0
direction = 0   # 0 = รอ hand up, 1 = รอ hand down
good_form = True
fixed_eye_line_y = None   # เก็บตำแหน่งเส้นนิ่งระดับตา

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # วาดเฉพาะ keypoints และเส้นเชื่อมที่ใช้จริง

        # รายการ keypoints ที่ใช้ (ตัด 2, 5, 0, 9, 10 ออก เหลือเฉพาะ 11,12,13,14,15,16,23,24,25,26,27,28)
        used_points = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        # วาดจุดทั้งหมดเป็นสีแดง
        for idx in used_points:
            cx = int(landmarks[idx].x * image.shape[1])
            cy = int(landmarks[idx].y * image.shape[0])
            cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)

        # วาดเส้นเชื่อมที่ใช้ใน logic
        # วาดเส้นปะสีน้ำเงิน
        def draw_dotted_line(a, b, color=(255, 0, 0), thickness=2, gap=8):
            ax = int(landmarks[a].x * image.shape[1])
            ay = int(landmarks[a].y * image.shape[0])
            bx = int(landmarks[b].x * image.shape[1])
            by = int(landmarks[b].y * image.shape[0])
            length = int(np.hypot(bx - ax, by - ay))
            for i in range(0, length, gap*2):
                start_x = int(ax + (bx - ax) * i / length)
                start_y = int(ay + (by - ay) * i / length)
                end_x = int(ax + (bx - ax) * min(i + gap, length) / length)
                end_y = int(ay + (by - ay) * min(i + gap, length) / length)
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)

        # วาดเส้นปะสีน้ำเงินสำหรับทุกเส้นเชื่อม
        draw_dotted_line(11, 12)
        draw_dotted_line(23, 24)
        draw_dotted_line(23, 25)
        draw_dotted_line(24, 26)
        draw_dotted_line(12, 14)
        draw_dotted_line(14, 16)
        draw_dotted_line(11, 13)
        draw_dotted_line(13, 15)
        if len(landmarks) > 28:
            draw_dotted_line(11, 23)
            draw_dotted_line(12, 24)
            draw_dotted_line(25, 27)
            draw_dotted_line(26, 28)

        # (1) หาตำแหน่งเส้นระดับตาใหม่: ระยะห่างของเส้น 9,10 กับ 12,11 แล้วหาร 2
        y_9 = landmarks[9].y
        y_10 = landmarks[10].y
        avg_y_9_10 = (y_9 + y_10) / 2
        y_12 = landmarks[12].y
        y_11 = landmarks[11].y
        avg_y_12_11 = (y_12 + y_11) / 2
        moving_eye_line_y = int(((avg_y_9_10 + avg_y_12_11) / 2) * image.shape[0])

        # (2) ตรวจตำแหน่งมือขวา/ซ้าย กับเส้น eye line
        left_wrist_y = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0])
        right_wrist_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image.shape[0])
        left_status = "Hand Up" if left_wrist_y < moving_eye_line_y else "Hand Down"
        right_status = "Hand Up" if right_wrist_y < moving_eye_line_y else "Hand Down"
        left_status_color = (0,255,0) if left_status == "Hand Up" else (0,0,255)
        right_status_color = (0,255,0) if right_status == "Hand Up" else (0,0,255)

        # (3) นับจำนวนครั้ง (count) - ถ้ามีข้างใดข้างหนึ่ง Hand Up (จาก Hand Down)
        # direction: 0 = รอ Hand Up, 1 = รอ Hand Down
        if (left_status == "Hand Up" or right_status == "Hand Up") and direction == 0:
            count += 1
            direction = 1
        elif (left_status == "Hand Down" and right_status == "Hand Down") and direction == 1:
            direction = 0

        # (4) คำนวณมุมแขนขวา (Shoulder-Elbow-Wrist)
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        feedback = "Correct" if 70 <= angle <= 110 else "Wrong"
        good_form = (70 <= angle <= 110)

        # (5) มุมหัวไหล่ซ้าย (14→12 กับ 12→24)
        l_shoulder = [landmarks[12].x, landmarks[12].y]
        l_elbow = [landmarks[14].x, landmarks[14].y]
        l_hip = [landmarks[24].x, landmarks[24].y]
        left_shoulder_angle = calculate_angle(l_elbow, l_shoulder, l_hip)

        # (6) มุมหัวไหล่ขวา (13→11 กับ 11→23)
        r_shoulder_2 = [landmarks[11].x, landmarks[11].y]
        r_elbow_2 = [landmarks[13].x, landmarks[13].y]
        r_hip = [landmarks[23].x, landmarks[23].y]
        right_shoulder_angle = calculate_angle(r_elbow_2, r_shoulder_2, r_hip)

        # (6.1) มุมที่จุด 13 (15-13-11)
        p15 = [landmarks[15].x, landmarks[15].y]
        p13 = [landmarks[13].x, landmarks[13].y]
        p11 = [landmarks[11].x, landmarks[11].y]
        angle_13 = calculate_angle(p15, p13, p11)

        # (6.2) มุมที่จุด 14 (12-14-16)
        p12 = [landmarks[12].x, landmarks[12].y]
        p14 = [landmarks[14].x, landmarks[14].y]
        p16 = [landmarks[16].x, landmarks[16].y]
        angle_14 = calculate_angle(p12, p14, p16)

        # (7) เช็คความขนาน หัวไหล่ (11-12) กับสะโพก (23-24)
        p11 = np.array([landmarks[11].x, landmarks[11].y])
        p12 = np.array([landmarks[12].x, landmarks[12].y])
        p23 = np.array([landmarks[23].x, landmarks[23].y])
        p24 = np.array([landmarks[24].x, landmarks[24].y])
        v_shoulder = p12 - p11
        v_hip = p24 - p23
        cos_theta = np.dot(v_shoulder, v_hip) / (np.linalg.norm(v_shoulder) * np.linalg.norm(v_hip) + 1e-8)
        angle_parallel = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
        parallel_status = "Parallel" if angle_parallel < 10 else "Not Parallel"

        # (7.1) เช็คความขนานของขา (23-25 กับ 24-26)
        p23_leg = np.array([landmarks[23].x, landmarks[23].y])
        p25_leg = np.array([landmarks[25].x, landmarks[25].y])
        p24_leg = np.array([landmarks[24].x, landmarks[24].y])
        p26_leg = np.array([landmarks[26].x, landmarks[26].y])
        v_leg_right = p25_leg - p23_leg
        v_leg_left = p26_leg - p24_leg
        cos_theta_leg = np.dot(v_leg_right, v_leg_left) / (np.linalg.norm(v_leg_right) * np.linalg.norm(v_leg_left) + 1e-8)
        angle_parallel_leg = np.arccos(np.clip(cos_theta_leg, -1.0, 1.0)) * 180 / np.pi
        parallel_status_leg = "Parallel" if angle_parallel_leg < 10 else "Not Parallel"

        # (8) แสดงผลบนจอ
        # ปรับระยะห่างและจัดกลุ่มข้อมูลให้เป็นระเบียบมากขึ้น
        y_base = 20
        y_step = 22
        x_left = 10
        font_scale_main = 0.7
        font_scale_sub = 0.55
        thickness_main = 2
        thickness_sub = 1

        # กลุ่มข้อมูลหลัก
        cv2.putText(image, f'Count: {count}', (x_left, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,255,0), thickness_main)

        # Logic สำหรับ Form
        form_text = ""
        form_color = (0,200,0)
        if left_status == "Hand Up" or right_status == "Hand Up":
            # ถ้าขาไม่ขนาน ให้ form เป็น Bad ทันที
            if parallel_status_leg != "Parallel":
                form_text = "Bad"
                form_color = (0,0,255)
            else:
                # เงื่อนไข parallel, arm normal, elbow ดีทั้งสองข้าง
                is_parallel = parallel_status == "Parallel"
                is_arm_normal = (left_shoulder_angle <= 125 and right_shoulder_angle <= 125)
                right_elbow_status = "Good elbow" if 130 <= angle_13 <= 150 else "Bad elbow"
                left_elbow_status = "Good elbow" if 130 <= angle_14 <= 150 else "Bad elbow"
                is_good_elbow = (right_elbow_status == "Good elbow" and left_elbow_status == "Good elbow")
                if is_parallel and is_arm_normal and is_good_elbow:
                    form_text = "Good"
                    form_color = (0,200,0)
                else:
                    form_text = "Bad"
                    form_color = (0,0,255)
        else:
            form_text = "Normal"
            form_color = (0,200,0)

        cv2.putText(image, f'Form: {form_text}', (x_left, y_base + y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, form_color, thickness_main)
        cv2.putText(image, f'Left: {left_status}', (x_left, y_base + 2*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, left_status_color, thickness_main)
        cv2.putText(image, f'Right: {right_status}', (x_left, y_base + 3*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, right_status_color, thickness_main)

        # กลุ่มข้อมูลองศา
        cv2.putText(image, f'L: {int(left_shoulder_angle)} deg', (x_left, y_base + 5*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255,128,0), thickness_sub)
        cv2.putText(image, f'R: {int(right_shoulder_angle)} deg', (x_left, y_base + 6*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (0,128,255), thickness_sub)
        cv2.putText(image, f'Shoulder-Hip: {parallel_status} ({angle_parallel:.1f} deg)', (x_left, y_base + 7*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255,255,0) if parallel_status=="Parallel" else (0,0,255), thickness_sub)

        # แสดงผลความขนานของขา
        # แสดงผลข้อความความขนานของขา
        leg_parallel_text = f'Legs: {parallel_status_leg} ({angle_parallel_leg:.1f} deg)'
        leg_parallel_color = (0,255,255) if parallel_status_leg=="Parallel" else (0,0,255)
        cv2.putText(image, leg_parallel_text, (x_left, y_base + 8*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, leg_parallel_color, thickness_main)

        # สถานะการกางแขน
        if left_shoulder_angle > 125 or right_shoulder_angle > 125:
            cv2.putText(image, 'Arm too wide!', (x_left, y_base + 9*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,0,255), 2)
        else:
            cv2.putText(image, 'Arm normal', (x_left, y_base + 9*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,200,0), 2)

        # มุมข้อศอก
        cv2.putText(image, f'Angle@13 (15-13-11): {int(angle_13)} deg', (x_left, y_base + 11*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255,200,0), thickness_sub)
        cv2.putText(image, f'Angle@14 (12-14-16): {int(angle_14)} deg', (x_left, y_base + 12*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (0,200,255), thickness_sub)

        # แสดง elbow status
        if left_status == "Hand Up" or right_status == "Hand Up":
            right_elbow_status = "Good elbow" if 130 <= angle_13 <= 155 else "Bad elbow"
            left_elbow_status = "Good elbow" if 130 <= angle_14 <= 155 else "Bad elbow"
            cv2.putText(image, f'Right: {right_elbow_status}', (x_left, y_base + 14*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,200,0) if right_elbow_status=="Good elbow" else (0,0,255), thickness_main)
            cv2.putText(image, f'Left: {left_elbow_status}', (x_left, y_base + 15*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,200,0) if left_elbow_status=="Good elbow" else (0,0,255), thickness_main)
        else:
            cv2.putText(image, 'Right: normal elbow', (x_left, y_base + 14*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,200,0), thickness_main)
            cv2.putText(image, 'Left: normal elbow', (x_left, y_base + 15*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,200,0), thickness_main)

    # (9) วาดเส้นระดับตา (ขยับตามจุด 12,11) ใช้เป็นเส้นสำหรับนับ (เส้นป่ะและบาง)
    eye_line_color = (255, 0, 255)  # สีม่วง
    eye_line_thickness = 1
    dash_length = 10
    x1, x2 = 0, image.shape[1]
    y = moving_eye_line_y
    for x in range(x1, x2, dash_length*2):
        x_end = min(x + dash_length, x2)
        cv2.line(image, (x, y), (x_end, y), eye_line_color, eye_line_thickness)

    cv2.imshow('Dumbbell Lateral Raise Counter', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()