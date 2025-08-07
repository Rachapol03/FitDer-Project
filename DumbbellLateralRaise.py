import streamlit as st
import tempfile
import numpy as np
import mediapipe as mp
import cv2

st.set_page_config(layout="wide")  # เพื่อให้จอใหญ่ 

# --- Main pose/analysis functions ---

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def draw_detailed_info_panel(image, count, left_status, right_status, left_shoulder_angle, 
                           right_shoulder_angle, parallel_status, angle_parallel, 
                           parallel_status_leg, angle_parallel_leg, angle_13, angle_14,
                           form_text, form_color, left_status_color, right_status_color,
                           direction):
    # Increase the height and width of the output panel for more info space and add spacing
    height, width = image.shape[:2]
    panel_width = 600  # much wider panel for more text space
    panel_spacing = 48  # more space between video and panel
    extra_height = 120  # Add extra vertical space for the info panel
    extended_image = np.zeros((height + extra_height, width + panel_spacing + panel_width, 3), dtype=np.uint8)
    extended_image[:height, :width] = image
    # Draw the info panel with spacing
    panel_x = width + panel_spacing
    cv2.rectangle(extended_image, (panel_x, 0), (panel_x + panel_width, height + extra_height), (40, 40, 40), -1)
    # Update all x_start and width + panel_width references below
    x_start = panel_x + 10
    # All references to (width + panel_width - 10) should be (panel_x + panel_width - 10)
    # shift all y positions down by 0 if needed, since info panel starts at y=0
    y_base = 25
    y_step = 22
    font_scale_main = 0.6
    font_scale_sub = 0.5
    thickness_main = 2
    thickness_sub = 1

    # MAIN STATUS
    cv2.rectangle(extended_image, (x_start, 10), (panel_x + panel_width - 10, 140), (70, 70, 70), 2)
    cv2.putText(extended_image, "MAIN STATUS", (x_start + 10, y_base), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(extended_image, f'Count: {count}', (x_start + 10, y_base + y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0, 255, 0), thickness_main)
    cv2.putText(extended_image, f'Form: {form_text}', (x_start + 10, y_base + 2*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, form_color, thickness_main)
    cv2.putText(extended_image, f'Left: {left_status}', (x_start + 10, y_base + 3*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, left_status_color, thickness_main)
    cv2.putText(extended_image, f'Right: {right_status}', (x_start + 10, y_base + 4*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, right_status_color, thickness_main)
    # SHOULDER ANGLES
    y_shoulder = y_base + 6*y_step
    cv2.rectangle(extended_image, (x_start, y_shoulder - 15), (panel_x + panel_width - 10, y_shoulder + 50), (70, 70, 70), 2)
    cv2.putText(extended_image, "SHOULDER ANGLES", (x_start + 10, y_shoulder), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(extended_image, f'Left: {int(left_shoulder_angle)} deg', (x_start + 10, y_shoulder + y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255, 128, 0), thickness_sub)
    cv2.putText(extended_image, f'Right: {int(right_shoulder_angle)} deg', (x_start + 10, y_shoulder + 2*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (0, 128, 255), thickness_sub)
    # ALIGNMENT
    y_parallel = y_shoulder + 4*y_step
    cv2.rectangle(extended_image, (x_start, y_parallel - 15), (panel_x + panel_width - 10, y_parallel + 70), (70, 70, 70), 2)
    cv2.putText(extended_image, "ALIGNMENT", (x_start + 10, y_parallel), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    shoulder_color = (255, 255, 0) if parallel_status == "Parallel" else (0, 0, 255)
    cv2.putText(extended_image, f'Shoulder-Hip: {parallel_status}', (x_start + 10, y_parallel + y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, shoulder_color, thickness_sub)
    cv2.putText(extended_image, f'({angle_parallel:.1f} deg)', (x_start + 10, y_parallel + 2*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, shoulder_color, thickness_sub)
    leg_color = (0, 255, 255) if parallel_status_leg == "Parallel" else (0, 0, 255)
    cv2.putText(extended_image, f'Legs: {parallel_status_leg}', (x_start + 10, y_parallel + 3*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, leg_color, thickness_sub)
    cv2.putText(extended_image, f'({angle_parallel_leg:.1f} deg)', (x_start + 10, y_parallel + 4*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, leg_color, thickness_sub)
    # ELBOW ANALYSIS
    y_elbow = y_parallel + 6*y_step
    cv2.rectangle(extended_image, (x_start, y_elbow - 15), (panel_x + panel_width - 10, y_elbow + 110), (70, 70, 70), 2)
    cv2.putText(extended_image, "ELBOW ANALYSIS", (x_start + 10, y_elbow), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # --- ELBOW ANALYSIS: 2 columns ---
    col_gap = 220
    # Left column (Angle@13)
    cv2.putText(extended_image, f'Angle@13 (R): {int(angle_13)} deg', (x_start + 10, y_elbow + y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255, 200, 0), thickness_sub)
    cv2.putText(extended_image, f'(15-13-11)', (x_start + 10, y_elbow + 2*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (180, 180, 180), 1)
    # Right column (Angle@14)
    cv2.putText(extended_image, f'Angle@14 (L): {int(angle_14)} deg', (x_start + col_gap, y_elbow + y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (0, 200, 255), thickness_sub)
    cv2.putText(extended_image, f'(12-14-16)', (x_start + col_gap, y_elbow + 2*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (180, 180, 180), 1)
    if left_status == "Hand Up" or right_status == "Hand Up":
        right_elbow_status = "Good" if 130 <= angle_13 <= 155 else "Bad"
        left_elbow_status = "Good" if 130 <= angle_14 <= 155 else "Bad"
        right_elbow_color = (0, 200, 0) if right_elbow_status == "Good" else (0, 0, 255)
        left_elbow_color = (0, 200, 0) if left_elbow_status == "Good" else (0, 0, 255)
    else:
        right_elbow_status = "Normal"
        left_elbow_status = "Normal"
        right_elbow_color = (0, 200, 0)
        left_elbow_color = (0, 200, 0)
    cv2.putText(extended_image, f'Right elbow: {right_elbow_status}', (x_start + 10, y_elbow + 5*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, right_elbow_color, thickness_sub)
    cv2.putText(extended_image, f'Left elbow: {left_elbow_status}', (x_start + 10, y_elbow + 6*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, left_elbow_color, thickness_sub)
    # ARM STATUS
    y_arm = y_elbow + 8*y_step
    cv2.rectangle(extended_image, (x_start, y_arm - 15), (panel_x + panel_width - 10, y_arm + 50), (70, 70, 70), 2)
    cv2.putText(extended_image, "ARM STATUS", (x_start + 10, y_arm), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if left_shoulder_angle > 125 or right_shoulder_angle > 125:
        arm_status = "Too Wide!"
        arm_color = (0, 0, 255)
    else:
        arm_status = "Normal"
        arm_color = (0, 200, 0)
    cv2.putText(extended_image, f'Arm: {arm_status}', (x_start + 10, y_arm + y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, arm_color, thickness_main)
    direction_text = "Waiting Up" if direction == 0 else "Waiting Down"
    cv2.putText(extended_image, f'Direction: {direction_text}', (x_start + 10, y_arm + 2*y_step),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255, 255, 255), thickness_sub)

    return extended_image

def draw_dotted_line(image, landmarks, a, b, color=(255, 0, 0), thickness=2, gap=8):
    ax = int(landmarks[a].x * image.shape[1])
    ay = int(landmarks[a].y * image.shape[0])
    bx = int(landmarks[b].x * image.shape[1])
    by = int(landmarks[b].y * image.shape[0])
    length = int(np.hypot(bx - ax, by - ay))
    if length == 0:
        return
    for i in range(0, length, gap*2):
        start_x = int(ax + (bx - ax) * i / length)
        start_y = int(ay + (by - ay) * i / length)
        end_x = int(ax + (bx - ax) * min(i + gap, length) / length)
        end_y = int(ay + (by - ay) * min(i + gap, length) / length)
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)

def run_dumbbell_lateral_raise_for_streamlit(source=0):
    cap = cv2.VideoCapture(source)
    count = 0
    direction = 0
    moving_eye_line_y = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (500, 700))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        left_status = "No Detection"
        right_status = "No Detection"
        left_shoulder_angle = 0
        right_shoulder_angle = 0
        parallel_status = "No Detection"
        angle_parallel = 0
        parallel_status_leg = "No Detection"
        angle_parallel_leg = 0
        angle_13 = 0
        angle_14 = 0
        form_text = "No Detection"
        form_color = (128, 128, 128)
        left_status_color = (128, 128, 128)
        right_status_color = (128, 128, 128)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # วาด keypoints
            used_points = [11,12,13,14,15,16,23,24,25,26,27,28]
            for idx in used_points:
                cx = int(landmarks[idx].x * image.shape[1])
                cy = int(landmarks[idx].y * image.shape[0])
                cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)

            # วาดเส้นประ
            connections = [(11,12),(23,24),(23,25),(24,26),(12,14),(14,16),
                          (11,13),(13,15),(11,23),(12,24),(25,27),(26,28)]
            for connection in connections:
                if len(landmarks) > max(connection):
                    draw_dotted_line(image, landmarks, connection[0], connection[1])

            # หาตำแหน่ง eye line
            y_9 = landmarks[9].y
            y_10 = landmarks[10].y
            avg_y_9_10 = (y_9 + y_10)/2
            y_12 = landmarks[12].y
            y_11 = landmarks[11].y
            avg_y_12_11 = (y_12 + y_11)/2
            moving_eye_line_y = int(((avg_y_9_10 + avg_y_12_11)/2) * image.shape[0])

            # ตำแหน่งข้อมือ
            left_wrist_y = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0])
            right_wrist_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image.shape[0])
            left_status = "Hand Up" if left_wrist_y < moving_eye_line_y else "Hand Down"
            right_status = "Hand Up" if right_wrist_y < moving_eye_line_y else "Hand Down"
            left_status_color = (0, 255, 0) if left_status == "Hand Up" else (0, 0, 255)
            right_status_color = (0, 255, 0) if right_status == "Hand Up" else (0, 0, 255)

            # Count reps
            if (left_status == "Hand Up" or right_status == "Hand Up") and direction == 0:
                count += 1
                direction = 1
            elif (left_status == "Hand Down" and right_status == "Hand Down") and direction == 1:
                direction = 0
            
            # คำนวณมุมหัวไหล่
            l_shoulder = [landmarks[12].x, landmarks[12].y]
            l_elbow = [landmarks[14].x, landmarks[14].y]
            l_hip = [landmarks[24].x, landmarks[24].y]
            left_shoulder_angle = calculate_angle(l_elbow, l_shoulder, l_hip)
            r_shoulder_2 = [landmarks[11].x, landmarks[11].y]
            r_elbow_2 = [landmarks[13].x, landmarks[13].y]
            r_hip = [landmarks[23].x, landmarks[23].y]
            right_shoulder_angle = calculate_angle(r_elbow_2, r_shoulder_2, r_hip)

            # มุมที่จุด 13 (15-13-11)
            p15 = [landmarks[15].x, landmarks[15].y]
            p13 = [landmarks[13].x, landmarks[13].y]
            p11 = [landmarks[11].x, landmarks[11].y]
            angle_13 = calculate_angle(p15, p13, p11)
            # มุมที่จุด 14 (12-14-16)
            p12 = [landmarks[12].x, landmarks[12].y]
            p14 = [landmarks[14].x, landmarks[14].y]
            p16 = [landmarks[16].x, landmarks[16].y]
            angle_14 = calculate_angle(p12, p14, p16)

            # เช็คขนานไหล่-สะโพก
            p11 = np.array([landmarks[11].x, landmarks[11].y])
            p12 = np.array([landmarks[12].x, landmarks[12].y])
            p23 = np.array([landmarks[23].x, landmarks[23].y])
            p24 = np.array([landmarks[24].x, landmarks[24].y])
            v_shoulder = p12 - p11
            v_hip = p24 - p23
            cos_theta = np.dot(v_shoulder, v_hip) / (np.linalg.norm(v_shoulder) * np.linalg.norm(v_hip) + 1e-8)
            angle_parallel = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
            parallel_status = "Parallel" if angle_parallel < 10 else "Not Parallel"

            # ขนานขา
            p23_leg = np.array([landmarks[23].x, landmarks[23].y])
            p25_leg = np.array([landmarks[25].x, landmarks[25].y])
            p24_leg = np.array([landmarks[24].x, landmarks[24].y])
            p26_leg = np.array([landmarks[26].x, landmarks[26].y])
            v_leg_right = p25_leg - p23_leg
            v_leg_left = p26_leg - p24_leg
            cos_theta_leg = np.dot(v_leg_right, v_leg_left) / (np.linalg.norm(v_leg_right) * np.linalg.norm(v_leg_left) + 1e-8)
            angle_parallel_leg = np.arccos(np.clip(cos_theta_leg, -1.0, 1.0)) * 180 / np.pi
            parallel_status_leg = "Parallel" if angle_parallel_leg < 10 else "Not Parallel"

            # ฟอร์ม
            form_text = ""
            form_color = (0,200,0)
            if left_status == "Hand Up" or right_status == "Hand Up":
                if parallel_status_leg != "Parallel":
                    form_text = "Bad"
                    form_color = (0,0,255)
                else:
                    is_parallel = parallel_status == "Parallel"
                    is_arm_normal = (left_shoulder_angle <= 125 and right_shoulder_angle <= 125)
                    right_elbow_status = "Good elbow" if 130 <= angle_13 <= 150 else "Bad elbow"
                    left_elbow_status = "Good elbow" if 130 <= angle_14 <= 150 else "Bad elbow"
                    is_good_elbow = (right_elbow_status == "Good elbow" and left_elbow_status == "Good elbow")
                    if is_parallel and is_arm_normal and is_good_elbow:
                        form_text = "Good"
                        form_color = (0, 200, 0)
                    else:
                        form_text = "Bad"
                        form_color = (0, 0, 255)
            else:
                form_text = "Normal"
                form_color = (0, 200, 0)

            # วาดเส้นระดับตา
            if moving_eye_line_y > 0:
                eye_line_color = (255, 0, 255)
                eye_line_thickness = 1
                dash_length = 10
                x1, x2 = 0, image.shape[1]
                y = moving_eye_line_y
                for x in range(x1, x2, dash_length*2):
                    x_end = min(x + dash_length, x2)
                    cv2.line(image, (x, y), (x_end, y), eye_line_color, eye_line_thickness)

        extended_image = draw_detailed_info_panel(
            image, count, left_status, right_status, left_shoulder_angle, 
            right_shoulder_angle, parallel_status, angle_parallel, 
            parallel_status_leg, angle_parallel_leg, angle_13, angle_14,
            form_text, form_color, left_status_color, right_status_color, direction
        )
        
        # สำหรับ streamlit ต้องแปลงเป็น RGB ก่อน
        yield cv2.cvtColor(extended_image, cv2.COLOR_BGR2RGB)
    
    cap.release()
    return

# --- Streamlit UI main ---
st.title("Dumbbell Lateral Raise Counter & Analyzer 👤💪")

stframe = st.empty()

mode = st.radio("เลือกแหล่งภาพ", ("อัปโหลดวิดีโอ", "เปิดกล้องเว็บแคม"))

uploaded_file = None
source = None
if mode == "อัปโหลดวิดีโอ":
    uploaded_file = st.file_uploader('อัปโหลดไฟล์วิดีโอ', type=['mp4', 'avi', 'mov', 'mkv'])

run_flag = st.button("เริ่มประมวลผล")

stframe = st.empty()

if run_flag:
    if mode == "อัปโหลดวิดีโอ":
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            source = temp_file.name
        else:
            st.warning("กรุณาอัปโหลดไฟล์วิดีโอก่อน")
    elif mode == "เปิดกล้องเว็บแคม":
        source = 0

    if source is not None:
        st.info("กำลังประมวลผล... กด STOP ที่ Streamlit เพื่อหยุด")
        for image in run_dumbbell_lateral_raise_for_streamlit(source):
            stframe.image(image, channels="RGB")
    else:
        st.write("เลือกรูปแบบแหล่งภาพและกด 'เริ่มประมวลผล'")
else:
    st.write("เลือกรูปแบบแหล่งภาพและกด 'เริ่มประมวลผล'")
