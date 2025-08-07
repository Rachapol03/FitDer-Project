
import streamlit as st

# --- Custom dark theme CSS ---
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #181818 !important;
        color: #f1f1f1 !important;
    }
    .stButton>button {
        background-color: #222 !important;
        color: #f1f1f1 !important;
        border: 1px solid #444 !important;
    }
    .stSelectbox, .stRadio, .stFileUploader, .stTextInput, .stTextArea {
        background-color: #222 !important;
        color: #f1f1f1 !important;
    }
    .st-bb, .st-c6, .st-cg, .st-cj, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
        background-color: #222 !important;
        color: #f1f1f1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('เลือกท่าออกกำลังกาย')

exercises = [
    'Dumbbell Lateral Raise',
    'Back Squat',
    'Dumbbell Thruster',
    'Side-Lying Leg Lift',
    'Floor Press',
    'Alternating Reverse Lunge',
    'Exercise 7',
    'Exercise 8',
    'Exercise 9'
]

selected = st.selectbox('เลือกท่า', exercises)


if selected == 'Dumbbell Lateral Raise':
    mode = st.radio('เลือกรูปแบบ', ['อัปโหลดวิดีโอ', 'เปิดกล้องเว็บแคม'])
    col1, col2 = st.columns([2, 2])
    if mode == 'อัปโหลดวิดีโอ':
        uploaded_file = col1.file_uploader('อัปโหลดไฟล์วิดีโอ', type=['mp4', 'avi', 'mov', 'mkv'])
        if uploaded_file is not None:
            process_btn = col1.button('ประมวลผล')
            if process_btn:
                from DumbbellLateralRaise4 import run_dumbbell_lateral_raise_for_streamlit
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                source = temp_file.name
                video_placeholder = col1.empty()
                for frame in run_dumbbell_lateral_raise_for_streamlit(source):
                    video_placeholder.image(frame, channels='RGB', use_container_width=True)
    else:
        webcam_btn = col1.button('เริ่มต้น')
        if webcam_btn:
            from DumbbellLateralRaise4 import run_dumbbell_lateral_raise_for_streamlit
            video_placeholder = col1.empty()
            for frame in run_dumbbell_lateral_raise_for_streamlit(0):
                video_placeholder.image(frame, channels='RGB', use_container_width=True)
else:
    if st.button('เริ่มต้น'):
        st.warning('ยังไม่ได้เชื่อมต่อโค้ดท่านี้')