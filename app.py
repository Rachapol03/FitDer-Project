import streamlit as st
import psycopg2
from RAG_Funtion import gen_res # สมมติว่า RAG_Funtion.py อยู่ใน PATH ที่ Streamlit เข้าถึงได้

# --- การตั้งค่าหน้า Streamlit ---
st.set_page_config(
    page_title="AI Chatbot โดย Gemini-like UI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- สีและสไตล์ (สามารถปรับแต่งเพิ่มเติมได้ใน CSS หรือโดยใช้ Streamlit theme) ---
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background-color: #F8F8F8; /* สีพื้นหลังอ่อนๆ */
    }
    .st-emotion-cache-1r6dm7m eczjsg311 { /* สำหรับข้อความผู้ใช้ */
        background-color: #E6F3FF;
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        align-self: flex-end;
    }
    .st-emotion-cache-1y4y1p7 eczjsg310 { /* สำหรับข้อความ AI */
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
        align-self: flex-start;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-ch5erd { /* Chat input container */
        border-top: 1px solid #EEEEEE;
        padding-top: 15px;
        padding-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- ส่วนหัวของแอป ---
st.title("🤖 Chatbot อัจฉริยะ")
st.markdown("สวัสดี! ฉันคือ Chatbot ที่พร้อมตอบคำถามของคุณ ลองถามอะไรก็ได้เลย!")

# --- การเชื่อมต่อฐานข้อมูล (ใช้ st.session_state เพื่อให้เชื่อมต่อครั้งเดียว) ---
if 'conn' not in st.session_state:
    try:
        st.session_state.conn = psycopg2.connect(
            dbname="fitderdb",
            user="admin",
            password="1234",
            host="localhost",
            port="5432"
        )
        # st.success("เชื่อมต่อฐานข้อมูลสำเร็จ!") # อาจจะซ่อนไว้เพื่อความสะอาดของ UI
    except Exception as e:
        st.error(f"ไม่สามารถเชื่อมต่อฐานข้อมูลได้: {e}")
        st.stop() # หยุดการทำงานหากเชื่อมต่อไม่ได้

# --- เก็บประวัติการสนทนา ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- แสดงประวัติการสนทนา ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ช่องป้อนข้อความสำหรับผู้ใช้ ---
if user_input := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
    # เพิ่มข้อความผู้ใช้ลงในประวัติ
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # แสดงผลการโหลดและเรียก AI
    with st.chat_message("assistant"):
        with st.spinner("AI กำลังคิดคำตอบ..."):
            try:
                # เรียกใช้ฟังก์ชัน gen_res จาก RAG_Funtion
                response_from_llm = gen_res(user_input, st.session_state.conn)
                st.markdown(response_from_llm)
                # เพิ่มข้อความ AI ลงในประวัติ
                st.session_state.messages.append({"role": "assistant", "content": response_from_llm})
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผลคำตอบ"})