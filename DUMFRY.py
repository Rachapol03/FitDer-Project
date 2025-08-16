import cv2
import mediapipe as mp
import numpy as np

import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(r'dum.mp4')
# cap = cv2.VideoCapture(0)

# --- Prepare CSV output ---


count = 0

# ---------- NEW: ตัวแปรสำหรับนับแบบ Standing -> NonStanding -> Standing ----------
# phase 0 = รอเริ่ม (ต้องอยู่ Standing ก่อน)
# phase 1 = อยู่ในรอบ (ออกจาก Standing แล้ว) รอกลับเข้า Standing เพื่อนับ +1
rep_phase = 0

# กันกระพริบ: ต้องยืนยันสถานะติดกัน N เฟรม
STANDING_CONFIRM = 3
NONSTANDING_CONFIRM = 3
standing_streak = 0
nonstanding_streak = 0
stable_main = None
prev_stable_main = None
# -------------------------------------------------------------------------------

good_form = True
moving_eye_line_y = None  # กัน NameError

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- เพิ่มตัวแปรเก็บมุมรอบก่อนหน้า (ของคุณเดิม) ---
prev_main_status = None
prev_main_angle = None


frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    # Calculate timestamp in seconds
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    timestamp_sec = frame_idx / fps

    frame = cv2.resize(frame, (640, 480))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ค่าเริ่มต้นสำหรับข้อความเมื่อยังไม่เจอ landmark
    left_status = right_status = "Hand Down"
    left_status_color = right_status_color = (0, 0, 255)
    left_shoulder_angle = right_shoulder_angle = 0
    angle_parallel = angle_parallel_leg = 0.0
    parallel_status = parallel_status_leg = "Not Parallel"
    angle_13 = angle_14 = 0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark


        # --- วาด keypoints ที่ใช้ (ยกเว้นจุด 0, 5 และ 2) ---
        used_points = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        skip_points = [5, 2, 0]
        for idx in used_points:
            if idx not in skip_points:
                cx = int(landmarks[idx].x * image.shape[1])
                cy = int(landmarks[idx].y * image.shape[0])
                cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)



        # --- เส้นปะสีเหลือง: ค่าเฉลี่ยความสูงของจุดที่ 5 กับ 2 ---
        dash_length_yellow = 10
        pt5_y = int(landmarks[5].y * image.shape[0])
        pt2_y = int(landmarks[2].y * image.shape[0])
        avg_5_2_y = int((pt5_y + pt2_y) / 2)
        for x in range(0, image.shape[1], dash_length_yellow*2):
            x_end = min(x + dash_length_yellow, image.shape[1])
            cv2.line(image, (x, avg_5_2_y), (x_end, avg_5_2_y), (0, 255, 255), 2)
        # สำหรับ logic เดิมที่ใช้ nose_y
        nose_y = int(landmarks[0].y * image.shape[0])  # ใช้จุด 0 สำหรับ logic เดิม

        hand_over_nose = False

        def draw_dotted_line(a, b, color=(255, 0, 0), thickness=2, gap=8):
            ax = int(landmarks[a].x * image.shape[1]); ay = int(landmarks[a].y * image.shape[0])
            bx = int(landmarks[b].x * image.shape[1]); by = int(landmarks[b].y * image.shape[0])
            length = int(np.hypot(bx - ax, by - ay))
            if length == 0: return
            for i in range(0, length, gap*2):
                sx = int(ax + (bx - ax) * i / length)
                sy = int(ay + (by - ay) * i / length)
                ex = int(ax + (bx - ax) * min(i + gap, length) / length)
                ey = int(ay + (by - ay) * min(i + gap, length) / length)
                cv2.line(image, (sx, sy), (ex, ey), color, thickness)

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

        # --- เส้นระดับตา (เฉลี่ย ตา/หัวไหล่) ---
        LE = mp_pose.PoseLandmark.LEFT_EYE.value
        RE = mp_pose.PoseLandmark.RIGHT_EYE.value
        LSH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        avg_eye_y = (landmarks[LE].y + landmarks[RE].y) / 2
        avg_sh_y  = (landmarks[LSH].y + landmarks[RSH].y) / 2
        moving_eye_line_y = int(((avg_eye_y + avg_sh_y) / 2) * image.shape[0])

        # --- สถานะข้อมือเทียบ eye line (ยังแสดงค่าได้ แต่จะไม่ใช้นับแล้ว) ---
        LW = mp_pose.PoseLandmark.LEFT_WRIST.value
        RW = mp_pose.PoseLandmark.RIGHT_WRIST.value
        left_wrist_y  = int(landmarks[LW].y * image.shape[0])
        right_wrist_y = int(landmarks[RW].y * image.shape[0])
        left_status  = "Hand Up" if left_wrist_y  < moving_eye_line_y else "Hand Down"
        right_status = "Hand Up" if right_wrist_y < moving_eye_line_y else "Hand Down"
        left_status_color  = (0,255,0) if left_status  == "Hand Up" else (0,0,255)
        right_status_color = (0,255,0) if right_status == "Hand Up" else (0,0,255)

        if left_wrist_y < nose_y or right_wrist_y < nose_y:
            hand_over_nose = True

        # --------- (REMOVED) นับแบบเดิมจาก Hand Up/Down ----------
        # if (left_status == "Hand Up" or right_status == "Hand Up") and direction == 0:
        #     count += 1
        #     direction = 1
        # elif (left_status == "Hand Down" and right_status == "Hand Down") and direction == 1:
        #     direction = 0
        # -----------------------------------------------------------

        # --- มุมแขนขวา (Shoulder-Elbow-Wrist) ---
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        feedback = "Correct" if 70 <= angle <= 110 else "Wrong"
        good_form = (70 <= angle <= 110)

        # --- มุมหัวไหล่ซ้าย/ขวา ---
        l_shoulder = [landmarks[12].x, landmarks[12].y]
        l_elbow = [landmarks[14].x, landmarks[14].y]
        l_hip = [landmarks[24].x, landmarks[24].y]
        left_shoulder_angle = calculate_angle(l_elbow, l_shoulder, l_hip)

        r_shoulder_2 = [landmarks[11].x, landmarks[11].y]
        r_elbow_2 = [landmarks[13].x, landmarks[13].y]
        r_hip = [landmarks[23].x, landmarks[23].y]
        right_shoulder_angle = calculate_angle(r_elbow_2, r_shoulder_2, r_hip)

        # --- มุมข้อศอก 13/14 ---
        p15 = [landmarks[15].x, landmarks[15].y]
        p13 = [landmarks[13].x, landmarks[13].y]
        p11 = [landmarks[11].x, landmarks[11].y]
        angle_13 = calculate_angle(p15, p13, p11)

        p12 = [landmarks[12].x, landmarks[12].y]
        p14 = [landmarks[14].x, landmarks[14].y]
        p16 = [landmarks[16].x, landmarks[16].y]
        angle_14 = calculate_angle(p12, p14, p16)

        # --- ความขนานหัวไหล่กับสะโพก ---
        p11_v = np.array([landmarks[11].x, landmarks[11].y])
        p12_v = np.array([landmarks[12].x, landmarks[12].y])
        p23_v = np.array([landmarks[23].x, landmarks[23].y])
        p24_v = np.array([landmarks[24].x, landmarks[24].y])
        v_shoulder = p12_v - p11_v
        v_hip = p24_v - p23_v
        cos_theta = np.dot(v_shoulder, v_hip) / (np.linalg.norm(v_shoulder) * np.linalg.norm(v_hip) + 1e-8)
        angle_parallel = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        parallel_status = "Parallel" if angle_parallel < 10 else "Not Parallel"

        # --- ความขนานของขา ---
        p23_leg = np.array([landmarks[23].x, landmarks[23].y])
        p25_leg = np.array([landmarks[25].x, landmarks[25].y])
        p24_leg = np.array([landmarks[24].x, landmarks[24].y])
        p26_leg = np.array([landmarks[26].x, landmarks[26].y])
        v_leg_right = p25_leg - p23_leg
        v_leg_left  = p26_leg - p24_leg
        cos_theta_leg = np.dot(v_leg_right, v_leg_left) / (np.linalg.norm(v_leg_right) * np.linalg.norm(v_leg_left) + 1e-8)
        angle_parallel_leg = np.degrees(np.arccos(np.clip(cos_theta_leg, -1.0, 1.0)))
        parallel_status_leg = "Parallel" if angle_parallel_leg < 10 else "Not Parallel"

        # ---------------- Main Status จากมุมหัวไหล่ (ของคุณเดิม) ----------------
        main_angle = max(left_shoulder_angle, right_shoulder_angle)
        if main_angle < 25:
            main_status = "Standing"
            main_status_color = (0, 255, 255)  # เหลือง
        elif main_angle < 90:
            main_status = "Moving"
            main_status_color = (255, 0, 0)    # น้ำเงิน
        else:
            main_status = "Arm Up"
            main_status_color = (0, 255, 0)    # เขียว

        # -------- NEW: ทำให้ main_status “นิ่ง” ก่อนใช้ตัดสิน (debounce) --------
        if main_status == "Standing":
            standing_streak += 1
            nonstanding_streak = 0
        else:
            nonstanding_streak += 1
            standing_streak = 0

        prev_stable_main = stable_main
        if standing_streak >= STANDING_CONFIRM:
            stable_main = "Standing"
        elif nonstanding_streak >= NONSTANDING_CONFIRM:
            stable_main = "NonStanding"  # รวม Moving + Arm Up
        # ถ้ายังไม่ถึงเกณฑ์ ให้คง stable_main เดิมไว้
        # -----------------------------------------------------------------------

        # -------- NEW: FSM สำหรับนับรอบ Standing -> NonStanding -> Standing ---
        if stable_main is not None:
            if rep_phase == 0:
                # ต้องเริ่มจาก Standing แล้วหลุดออกไป
                if prev_stable_main == "Standing" and stable_main == "NonStanding":
                    rep_phase = 1
            elif rep_phase == 1:
                # กลับเข้า Standing = ครบรอบ
                if stable_main == "Standing":
                    count += 1
                    rep_phase = 0
        # -----------------------------------------------------------------------

        # --- Logic สำหรับ Form (ของคุณเดิม ปรับเล็กน้อยให้ทำงานเหมือนเดิม) ---
        y_base = 20; y_step = 22; x_left = 10
        font_scale_main = 0.7; font_scale_sub = 0.55
        thickness_main = 2; thickness_sub = 1

        cv2.putText(image, f'Count: {count}', (x_left, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,255,0), thickness_main)

        form_text = ""; form_color = (0,200,0)
        if abs(left_shoulder_angle - right_shoulder_angle) > 20:
            form_text = "Bad"; form_color = (0,0,255)
        elif 'main_status' in locals() and main_status == "Standing":
            if left_shoulder_angle > 25 or right_shoulder_angle > 25:
                form_text = "Bad"; form_color = (0,0,255)
            else:
                form_text = "Normal"; form_color = (0,200,0)
        # ลบเงื่อนไข hand_over_nose ออก ไม่ตัดสิน Bad จากยกข้อมือสูงกว่าเส้นปะที่ 5 กับ 2
        elif left_status == "Hand Up" or right_status == "Hand Up":
            if parallel_status_leg != "Parallel":
                form_text = "Bad"; form_color = (0,0,255)
            else:
                is_parallel = (parallel_status == "Parallel")
                is_arm_normal = (left_shoulder_angle <= 125 and right_shoulder_angle <= 125)
                right_elbow_status = "Good elbow" if 130 <= angle_13 <= 150 else "Bad elbow"
                left_elbow_status  = "Good elbow" if 130 <= angle_14 <= 150 else "Bad elbow"
                is_good_elbow = (right_elbow_status == "Good elbow" and left_elbow_status == "Good elbow")
                form_text = "Good" if (is_parallel and is_arm_normal and is_good_elbow) else "Bad"
                form_color = (0,200,0) if form_text=="Good" else (0,0,255)
        else:
            form_text = "Normal"; form_color = (0,200,0)



        cv2.putText(image, f'Form: {form_text}', (x_left, y_base + y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, form_color, thickness_main)
        cv2.putText(image, f'Left: {left_status}', (x_left, y_base + 2*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, left_status_color, thickness_main)
        cv2.putText(image, f'Right: {right_status}', (x_left, y_base + 3*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, right_status_color, thickness_main)

        cv2.putText(image, f'Status: {main_status}', (x_left, y_base + 4*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, main_status_color, thickness_main)

        # ตรวจ Standing->Moving ลดมุมแล้ว Bad (ตามโค้ดเดิม)
        if prev_main_status == "Standing" and main_status == "Moving":
            if prev_main_angle is not None and main_angle < prev_main_angle:
                form_text = "Bad"; form_color = (0,0,255)

        prev_main_status = main_status
        prev_main_angle = main_angle

        cv2.putText(image, f'L: {int(left_shoulder_angle)} deg', (x_left, y_base + 5*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255,128,0), thickness_sub)
        cv2.putText(image, f'R: {int(right_shoulder_angle)} deg', (x_left, y_base + 6*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (0,128,255), thickness_sub)
        cv2.putText(image, f'Shoulder-Hip: {parallel_status} ({angle_parallel:.1f} deg)', (x_left, y_base + 7*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub,
                    (255,255,0) if parallel_status=="Parallel" else (0,0,255), thickness_sub)
        leg_parallel_text = f'Legs: {parallel_status_leg} ({angle_parallel_leg:.1f} deg)'
        leg_parallel_color = (0,255,255) if parallel_status_leg=="Parallel" else (0,0,255)
        cv2.putText(image, leg_parallel_text, (x_left, y_base + 8*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, leg_parallel_color, thickness_main)

        if left_shoulder_angle > 125 or right_shoulder_angle > 125:
            cv2.putText(image, 'Arm too wide!', (x_left, y_base + 9*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,0,255), 2)
        else:
            cv2.putText(image, 'Arm normal', (x_left, y_base + 9*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,200,0), 2)

        cv2.putText(image, f'Angle@13 (15-13-11): {int(angle_13)} deg', (x_left, y_base + 11*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255,200,0), thickness_sub)
        cv2.putText(image, f'Angle@14 (12-14-16): {int(angle_14)} deg', (x_left, y_base + 12*y_step),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (0,200,255), thickness_sub)

        if left_status == "Hand Up" or right_status == "Hand Up":
            right_elbow_status = "Good elbow" if 130 <= angle_13 <= 155 else "Bad elbow"
            left_elbow_status  = "Good elbow" if 130 <= angle_14 <= 155 else "Bad elbow"
            cv2.putText(image, f'Right: {right_elbow_status}', (x_left, y_base + 14*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main,
                        (0,200,0) if right_elbow_status=="Good elbow" else (0,0,255), thickness_main)
            cv2.putText(image, f'Left: {left_elbow_status}', (x_left, y_base + 15*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main,
                        (0,200,0) if left_elbow_status=="Good elbow" else (0,0,255), thickness_main)
        else:
            cv2.putText(image, 'Right: normal elbow', (x_left, y_base + 14*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,200,0), thickness_main)
            cv2.putText(image, 'Left: normal elbow', (x_left, y_base + 15*y_step),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0,200,0), thickness_main)

    # --- วาดเส้นระดับตา ---
    if moving_eye_line_y is not None:
        eye_line_color = (255, 0, 255)
        eye_line_thickness = 1
        dash_length = 10
        y = moving_eye_line_y
        for x in range(0, image.shape[1], dash_length*2):
            x_end = min(x + dash_length, image.shape[1])
            cv2.line(image, (x, y), (x_end, y), eye_line_color, eye_line_thickness)

    cv2.imshow('Dumbbell Lateral Raise Counter', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
