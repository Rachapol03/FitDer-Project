
import av
import cv2
import time
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import mediapipe as mp

st.set_page_config(page_title="Workout Detection", page_icon="💪", layout="wide")

# ---------- defaults in session ----------
if "SRC" not in st.session_state:
    st.session_state["SRC"] = "ไฟล์วิดีโอ"
if "CAM_MODE" not in st.session_state:
    st.session_state["CAM_MODE"] = "Front"
if "LATEST_STATS" not in st.session_state:
    st.session_state["LATEST_STATS"] = (0, "Normal", "Hand Down", "Hand Down")
if "RUN_VIDEO" not in st.session_state:
    st.session_state["RUN_VIDEO"] = False
if "VIDEO_BYTES" not in st.session_state:
    st.session_state["VIDEO_BYTES"] = None
# ธีมเริ่มต้น
if "THEME" not in st.session_state:
    st.session_state["THEME"] = "Light"   # หรือ "Dark" ถ้าอยากเริ่มมืด

# ===== Theme-aware CSS =====
def theme_css(theme: str) -> str:
    dark = theme == "Dark"

    # ----- YouTube-like dark palette -----
    bg        = "#0f0f0f" if dark else "#ffffff"   # page bg
    surface   = "#181818" if dark else "#ffffff"   # header / sections
    card_bg   = "#212121" if dark else "#ffffff"   # cards
    border    = "rgba(255,255,255,0.08)" if dark else "#e5e7eb"
    text      = "#e6e6e6" if dark else "#111827"
    subtext   = "rgba(255,255,255,0.65)" if dark else "#6b7280"
    shadow    = "0 1px 2px rgba(0,0,0,0.5)" if dark else "0 1px 2px rgba(0,0,0,0.06)"

  # badges
    good_bg, good_fg = ("#1b3c2d", "#86efac") if dark else ("#dcfce7", "#166534")
    bad_bg,  bad_fg  = ("#3a1a1a", "#fecaca") if dark else ("#fee2e2", "#991b1b")
    norm_bg, norm_fg = ("#2d2a16", "#f0e68c") if dark else ("#fef9c3", "#854d0e")

    # buttons / controls (ปุ่มบนซ้าย/Run/วิทยุ/อัปโหลด ฯลฯ)
    btn_bg     = "#2a2a2a" if dark else "#11182708"
    btn_bg_hov = "#343434" if dark else "#11182710"
    btn_fg     = "#f5f5f5" if dark else "#111827"
    radio_dot  = "#f1f1f1" if dark else "#111827"

    return f"""
    <style>
      * {{ box-sizing:border-box; margin:0; padding:0; -webkit-tap-highlight-color:transparent; }}
      :root, .stApp {{
        background:{bg} !important; color:{text} !important;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      }}
      .block-container {{ padding-top:8px !important; max-width:1400px; }}

      /* Header */
      .header {{
        height:60px; display:flex; align-items:center; justify-content:space-between;
        background:{surface}; border:1px solid {border}; border-radius:12px; padding:8px 12px;
        position:sticky; top:0; z-index:10; backdrop-filter:blur(10px); box-shadow:{shadow};
      }}
      .header-title {{ font-size:22px; font-weight:800; color:{text}; }}

      /* Cards */
      .stat-card {{
        background:{card_bg}; border:1px solid {border}; border-radius:14px; padding:16px;
        display:flex; flex-direction:column; gap:8px; box-shadow:{shadow};
      }}
      .stat-label {{ font-size:12px; color:{subtext}; font-weight:600; }}
      .stat-value {{ font-size:28px; font-weight:800; color:{text}; line-height:1; }}
      .stat-badge {{ font-size:14px; font-weight:700; padding:6px 10px; border-radius:999px; display:inline-block; }}
      .badge-good {{ background:{good_bg}; color:{good_fg}; }}
      .badge-bad  {{ background:{bad_bg};  color:{bad_fg};  }}
      .badge-normal {{ background:{norm_bg}; color:{norm_fg}; }}

      /* Streamlit controls -> ให้มืดตามธีม */
      .stButton>button {{
        background:{btn_bg}; color:{btn_fg}; border:1px solid {border}; border-radius:10px;
      }}
      .stButton>button:hover {{ background:{btn_bg_hov}; }}
      .stDownloadButton>button {{ background:{btn_bg}; color:{btn_fg}; border:1px solid {border}; }}

      .stRadio [data-baseweb="radio"]>label>div:first-child {{
        border-color:{border}; background:{card_bg};
      }}
      .stRadio [role="radio"][aria-checked="true"]>div:first-child {{
        border-color:{radio_dot}; box-shadow:0 0 0 3px rgba(255,255,255,0.08);
      }}

      .stFileUploader, .stFileUploader > div {{ color:{text}; }}
      .stFileUploader>div>div {{
        background:{surface}; border:1px solid {border}; border-radius:12px;
      }}

      .stExpander, .stExpander>details {{
        background:{surface}; border:1px solid {border}; border-radius:12px;
      }}

      /* WebRTC video */
      video {{ border-radius:14px; border:1px solid {border}; }}
      video::-webkit-media-controls {{ display:none !important; }}
    </style>
    """


st.markdown(theme_css(st.session_state["THEME"]), unsafe_allow_html=True)

# ===== Header + Theme Toggle (single button) =====
# ใช้ค่าธีมจาก session (มีตั้งไว้ด้านบนแล้ว)
current_theme = st.session_state.get("THEME", "Light")
st.markdown(theme_css(current_theme), unsafe_allow_html=True)
st.markdown('<div class="header"><div class="header-title"></div>', unsafe_allow_html=True)

h1, h2 = st.columns([1, 1], vertical_alignment="center")
with h1:
    st.markdown('<div class="header"><div class="header-title">Dumbbell Lateral Raise</div>', unsafe_allow_html=True)

with h2:
    if current_theme == "Light":
        label, next_theme = "🌙  Dark mode", "Dark"
    else:
        label, next_theme = "☀️  Light mode", "Light"

    if st.button(label, use_container_width=True):
        st.session_state["THEME"] = next_theme
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# ===== Helpers =====
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def letterbox_16_9(img, target_w=1280, target_h=720, bg_gray=255):
    """บังคับ 16:9 พร้อมพื้นหลังตามธีม (ขาว=255, ดำ=0)"""
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    src_ratio = w / h
    if src_ratio > target_ratio:
        new_w = target_w; new_h = int(new_w / src_ratio)
    else:
        new_h = target_h; new_w = int(new_h * src_ratio)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((target_h, target_w, 3), bg_gray, dtype=np.uint8)
    y0 = (target_h - new_h) // 2; x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

# ===== Processor =====
class LateralRaiseProcessor(VideoProcessorBase):
    def __init__(self, bg_gray=255):
        self.pose = mp_pose.Pose()
        self.count = 0
        self.direction = 0
        self.form_text = "Normal"
        self.left_status = "Hand Down"
        self.right_status = "Hand Down"
        self.bg_gray = bg_gray  # พื้นหลัง 16:9 ตามธีม


    def _process_logic(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image = image_bgr

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            used_points = [11,12,13,14,15,16,23,24,25,26,27,28]
            for idx in used_points:
                cx = int(lm[idx].x * image.shape[1]); cy = int(lm[idx].y * image.shape[0])
                cv2.circle(image, (cx, cy), 6, (0,0,255), -1)

            def dotted(a, b, color=(255,0,0), thickness=2, gap=8):
                ax = int(lm[a].x * image.shape[1]); ay = int(lm[a].y * image.shape[0])
                bx = int(lm[b].x * image.shape[1]); by = int(lm[b].y * image.shape[0])
                length = int(np.hypot(bx-ax, by-ay))
                if length <= 0: return
                for i in range(0, length, gap*2):
                    sx = int(ax + (bx-ax)*i/length); sy = int(ay + (by-ay)*i/length)
                    ex = int(ax + (bx-ax)*min(i+gap, length)/length); ey = int(ay + (by-ay)*min(i+gap, length)/length)
                    cv2.line(image, (sx,sy), (ex,ey), color, thickness)

            for a,b in [(11,12),(23,24),(23,25),(24,26),(12,14),(14,16),(11,13),(13,15),(11,23),(12,24),(25,27),(26,28)]:
                dotted(a,b)

            y_eye = int((((lm[9].y+lm[10].y)/2 + (lm[12].y+lm[11].y)/2)/2) * image.shape[0])
            for x in range(0, image.shape[1], 20):
                cv2.line(image, (x, y_eye), (min(x+10, image.shape[1]), y_eye), (255,0,255), 1)

            lw_y = int(lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0])
            rw_y = int(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image.shape[0])
            self.left_status  = "Hand Up" if lw_y < y_eye else "Hand Down"
            self.right_status = "Hand Up" if rw_y < y_eye else "Hand Down"

            if (self.left_status == "Hand Up" or self.right_status == "Hand Up") and self.direction == 0:
                self.count += 1; self.direction = 1
            elif (self.left_status == "Hand Down" and self.right_status == "Hand Down") and self.direction == 1:
                self.direction = 0


            l_sh=[lm[12].x,lm[12].y]; l_el=[lm[14].x,lm[14].y]; l_hip=[lm[24].x,lm[24].y]
            r_sh2=[lm[11].x,lm[11].y]; r_el2=[lm[13].x,lm[13].y]; r_hip=[lm[23].x,lm[23].y]
            left_sh_angle  = calculate_angle(l_el, l_sh, l_hip)
            right_sh_angle = calculate_angle(r_el2, r_sh2, r_hip)

            p15=[lm[15].x,lm[15].y]; p13=[lm[13].x,lm[13].y]; p11=[lm[11].x,lm[11].y]
            ang13 = calculate_angle(p15,p13,p11)
            p12=[lm[12].x,lm[12].y]; p14=[lm[14].x,lm[14].y]; p16=[lm[16].x,lm[16].y]
            ang14 = calculate_angle(p12,p14,p16)

            P11=np.array([lm[11].x,lm[11].y]); P12=np.array([lm[12].x,lm[12].y])
            P23=np.array([lm[23].x,lm[23].y]); P24=np.array([lm[24].x,lm[24].y])
            v_sh=P12-P11; v_hip=P24-P23
            cos_t = np.dot(v_sh, v_hip)/(np.linalg.norm(v_sh)*np.linalg.norm(v_hip)+1e-8)
            angle_parallel = np.degrees(np.arccos(np.clip(cos_t,-1,1)))
            parallel = (angle_parallel < 10)

            v_lr = np.array([lm[25].x-lm[23].x, lm[25].y-lm[23].y])
            v_ll = np.array([lm[26].x-lm[24].x, lm[26].y-lm[24].y])
            cos_leg = np.dot(v_lr, v_ll)/(np.linalg.norm(v_lr)*np.linalg.norm(v_ll)+1e-8)
            angle_leg = np.degrees(np.arccos(np.clip(cos_leg,-1,1)))
            parallel_leg = (angle_leg < 10)

            if (self.left_status=="Hand Up" or self.right_status=="Hand Up"):
                if not parallel_leg:
                    self.form_text = "Bad"
                else:
                    arm_normal = (left_sh_angle<=125 and right_sh_angle<=125)
                    elbow_ok = (130<=ang13<=150) and (130<=ang14<=150)
                    self.form_text = "Good" if (parallel and arm_normal and elbow_ok) else "Bad"
            else:
                self.form_text = "Normal"

        image = cv2.flip(image, 1)
        image = letterbox_16_9(image, 1280, 720, bg_gray=self.bg_gray)
        return image

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        out = self._process_logic(img)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

# ===== Layout: วิดีโอซ้าย / กล่องสรุปขวา =====
webrtc_ctx = None
col_left, col_right = st.columns([2, 1])

# ---------- RIGHT: stats boxes ----------
with col_right:
    st.markdown("#### สรุปแบบเรียลไทม์")
    box_count = st.empty(); box_form = st.empty(); box_state = st.empty()

    def render_stats(count, form_text, lstat, rstat):
        badge = ('<span class="stat-badge badge-good">Good</span>' if form_text=="Good"
                 else '<span class="stat-badge badge-bad">Bad</span>' if form_text=="Bad"
                 else '<span class="stat-badge badge-normal">Normal</span>')
        box_count.markdown(
            f'<div class="stat-card"><div class="stat-label">จำนวนครั้ง</div>'
            f'<div class="stat-value">{count}</div></div>', unsafe_allow_html=True)
        box_form.markdown(
            f'<div class="stat-card"><div class="stat-label">ฟอร์ม</div>{badge}</div>',
            unsafe_allow_html=True)
        box_state.markdown(
            f'<div class="stat-card"><div class="stat-label">สถานะ</div>'
            f'<div class="stat-value" style="font-size:18px;">Left: {lstat} | Right: {rstat}</div></div>',
            unsafe_allow_html=True)

    c, f, ls, rs = st.session_state["LATEST_STATS"]
    render_stats(c, f, ls, rs)

# ---------- LEFT: video area + controls under it ----------
with col_left:
    video_view = st.empty()  # สำหรับโหมดไฟล์
    src = st.session_state["SRC"]
    cam_mode = st.session_state["CAM_MODE"]
    dark = st.session_state["THEME"] == "Dark"
    bg_gray = 24 if dark else 255 

    # กล้อง (แสดงทันที)
    if src == "กล้อง":
        facing = "user" if cam_mode == "Front" else "environment"
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_ctx = webrtc_streamer(
            key="workout-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": {"facingMode": facing}, "audio": False},
            # ส่ง bg_gray เข้า processor ให้พื้น 16:9 ตรงธีม
            video_processor_factory=lambda: LateralRaiseProcessor(bg_gray=bg_gray),
            async_processing=True,
            video_html_attrs={
                "controls": False, "muted": True, "playsinline": True,
                "style": "width:100%;height:auto;border-radius:14px;"
            },
        )


    # แผงควบคุมใต้จอ
    with st.expander("🎛️ ตั้งค่าวิดีโอ / กล้อง", expanded=False):
        src_pick = st.radio("เลือกแหล่งวิดีโอ", ["กล้อง", "ไฟล์วิดีโอ"],
                            horizontal=True, index=0 if src=="กล้อง" else 1)
        st.session_state["SRC"] = src_pick

        if src_pick == "กล้อง":
            cam_pick = st.radio("สลับกล้อง", ["Front", "Back"], horizontal=True,
                                index=0 if cam_mode=="Front" else 1)
            st.session_state["CAM_MODE"] = cam_pick
            st.info("เปิดกล้องแล้วด้านบน")
        else:
            uploaded = st.file_uploader("อัปโหลด .mp4 (บังคับ 16:9 อัตโนมัติ)", type=["mp4"])
            run = st.button("▶️ Run video")
            if uploaded and run:
                st.session_state["RUN_VIDEO"] = True
                st.session_state["VIDEO_BYTES"] = uploaded.getvalue()

# ---------- FILE MODE: realtime update ----------
if st.session_state["SRC"] == "ไฟล์วิดีโอ" and st.session_state["RUN_VIDEO"]:
    st.session_state["RUN_VIDEO"] = False
    bytes_data = st.session_state["VIDEO_BYTES"]
    if bytes_data:
        tmp_path = f"./_tmp_{int(time.time())}.mp4"
        with open(tmp_path, "wb") as f:
            f.write(bytes_data)

        dark = st.session_state["THEME"] == "Dark"
        proc = LateralRaiseProcessor(bg_gray=(24 if dark else 255))
        cap = cv2.VideoCapture(tmp_path)
        delay = 1.0 / max(24.0, cap.get(cv2.CAP_PROP_FPS) or 24.0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = proc._process_logic(frame)
            video_view.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)
            render_stats(proc.count, proc.form_text, proc.left_status, proc.right_status)
            time.sleep(delay)

        cap.release()
        st.session_state["LATEST_STATS"] = (proc.count, proc.form_text, proc.left_status, proc.right_status)
        st.success(f"เสร็จสิ้น! นับได้ {proc.count} ครั้ง")

# ---------- WEBCAM MODE: pull realtime stats ----------
if st.session_state["SRC"] == "กล้อง":
    st.autorefresh(interval=300, key="rt")
    if webrtc_ctx and webrtc_ctx.state.playing:
        vp = webrtc_ctx.video_processor
        if vp is not None:
            st.session_state["LATEST_STATS"] = (vp.count, vp.form_text, vp.left_status, vp.right_status)
            c, f, ls, rs = st.session_state["LATEST_STATS"]
            # วาดซ้ำตอน rerun
            with col_right:
                pass  # layout รักษาไว้

