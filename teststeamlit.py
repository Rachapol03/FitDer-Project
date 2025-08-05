import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# เตรียม mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ตัวแปรสำหรับนับจำนวนครั้งและสถานะ
counter = 0
stage = None

# ฟังก์ชันคำนวณองศาระหว่างจุด 3 จุด
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_frame_for_lateral_raise(frame):
    global counter, stage
    
    # พลิกภาพให้เหมือนมองกระจก
    frame = cv2.flip(frame, 1)

    # ประมวลผลภาพด้วย MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7).process(image)
    
    landmarks_data = []
    connections_data = []
    feedback = "กำลังรอ"

    if results.pose_landmarks:
        # ดึงจุด Landmark ที่สำคัญ
        landmarks = results.pose_landmarks.landmark
        
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # คำนวณองศา
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # ตรรกะการนับ Dumbbell Lateral Raise
        if 70 < angle < 110:
            feedback = "ถูกต้อง"
            stage = "ขึ้น"
        elif angle < 30:
            if stage == "ขึ้น":
                stage = "ลง"
                counter += 1
                
        # สร้างข้อมูลสำหรับวาดจุดและเส้นบน Canvas
        for landmark in landmarks:
            landmarks_data.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })

        for connection in mp_pose.POSE_CONNECTIONS:
            connections_data.append(connection)

    # ส่งข้อมูลกลับไปที่ Frontend
    return {
        'rep_count': counter,
        'form_status': feedback,
        'pose_state': stage if stage else "กำลังรอ",
        'landmarks': landmarks_data,
        'connections': connections_data
    }

@app.route('/')
def index():
    # หน้าหลักที่จะแสดงผล HTML
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ประมวลผลเฟรมด้วยโค้ดของคุณ
    processed_data = process_frame_for_lateral_raise(frame)
    
    # ส่งข้อมูลที่ประมวลผลแล้วกลับไป
    socketio.emit('workout_data', processed_data)
    socketio.emit('pose_landmarks', processed_data)

if __name__ == '__main__':
    socketio.run(app, debug=True)