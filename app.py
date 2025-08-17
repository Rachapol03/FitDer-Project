from flask import Flask, request, jsonify
from flask_cors import CORS
from RFuntion import gen_res_mongodb, close_mongodb_connection

app = Flask(__name__)
CORS(app) # อนุญาตให้ Front-end เรียกใช้งานได้ (CORS)
@app.get("/")
# Endpoint สำหรับการสนทนา
@app.route('/chat', methods=['POST'])
def chat():
    """
    รับข้อความจาก Front-end, ประมวลผลด้วย RAG และ Gemini,
    แล้วส่งคำตอบกลับ
    """
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({'response': 'กรุณาพิมพ์ข้อความ'})

    # เรียกใช้ฟังก์ชันจาก RFuntion.py เพื่อรับคำตอบจาก Gemini
    try:
        bot_response = gen_res_mongodb(user_message)
        return jsonify({'response': bot_response})
    except Exception as e:
        # กรณีเกิดข้อผิดพลาดในการประมวลผล
        print(f"Error processing chat message: {e}")
        return jsonify({'response': 'ขออภัย เกิดข้อผิดพลาดในการตอบคำถาม กรุณาลองใหม่อีกครั้งค่ะ'})

if __name__ == '__main__':
    # เมื่อปิดโปรแกรม ให้ปิดการเชื่อมต่อ MongoDB ด้วย
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        # เรียกฟังก์ชันปิดการเชื่อมต่อ MongoDB เมื่อเซิร์ฟเวอร์หยุดทำงาน
        close_mongodb_connection()