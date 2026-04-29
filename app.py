import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import tempfile
import sys
import base64
import collections
import ast
import plotly.express as px
import pandas as pd
from io import BytesIO
import time

# ====================== 登录验证 ======================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def check_login(username, password):
    return username == "admin" and password == "123456"


if not st.session_state.logged_in:
    st.title("🔒 管理员登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.success("登录成功！")
            st.rerun()
        else:
            st.error("用户名或密码错误")
    st.stop()

# ====================== 风险等级配置 ======================
RISK = {
    "smoke": "高危",
    "smoking": "高危",
    "phone": "中危",
    "cellphone": "中危",
    "fall": "高危",
    "falling": "高危",
    "head": "高危",
    "normal": "低危",
    "hat": "低危"
}

COLOR = {
    "高危": "red",
    "中危": "orange",
    "低危": "green"
}


# ====================== 图片转base64 ======================
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("background.png")

# ====================== 页面样式 ======================
st.set_page_config(page_title="施工安全检测系统", layout="wide")
st.markdown(f"""
<style>
    .main-title {{text-align:center; font-size:32px; font-weight:bold; margin-bottom:10px;}}
    .sub-title {{text-align:center; font-size:18px; color:#666; margin-bottom:20px;}}
    .stApp {{
        background-image: url("data:image/png;base64,{img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏗️ 基于YOLOv8 工地安防智能监测系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">实时摄像头 | 图片检测 | 视频检测 | 区域警戒 | 违规统计</div>',
            unsafe_allow_html=True)


# ====================== 路径 ======================
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


BASE_PATH = get_base_path()
MODEL_PATH = os.path.join(BASE_PATH, "moxing", "weights", "best.pt")
ALARM_FOLDER = os.path.join(BASE_PATH, "alarm_img")
os.makedirs(ALARM_FOLDER, exist_ok=True)
log_file = os.path.join(BASE_PATH, "detection_log.txt")


# ====================== 模型全局缓存 只加载一次 ======================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()
class_names = model.names

# ====================== 功能菜单 ======================
task_type = st.radio(
    "请选择功能模式",
    ["📹 实时摄像头检测", "🖼️ 图片检测", "🎞️ 视频检测", "📊 违规统计管理系统"],
    horizontal=True
)

# ====================== 摄像头检测｜大屏+流畅｜无语音 ======================
if task_type == "📹 实时摄像头检测":
    col_control, col_video = st.columns([1, 3])
    with col_control:
        st.subheader("⚙️ 系统参数配置")
        conf_threshold = st.slider("检测置信度", 0.1, 0.9, 0.5)
        capture_interval = st.selectbox("违规抓拍间隔(秒)", [5, 10, 15, 20], index=1)

        st.subheader("⚙️ 警戒区域")
        x1 = st.slider("X1", 0, 1280, 150)
        y1 = st.slider("Y1", 0, 720, 100)
        x2 = st.slider("X2", 0, 1280, 1000)
        y2 = st.slider("Y2", 0, 720, 600)

        run = st.checkbox("开启摄像头")
        st.subheader("🚨 告警状态")
        alert_info = st.empty()
        st.subheader("📊 实时识别统计")
        stat_info = st.empty()

    with col_video:
        frame_show = st.empty()

    cap = None
    last_save_time = 0

    if run:
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS,20)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                frame,
                conf=conf_threshold,
                verbose=False,
                imgsz=640
            )
            res_frame = results[0].plot()
            cv2.rectangle(res_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            total_persons = 0
            detected_classes = []
            warn_in_area = False
            high_risk = False

            for box in results[0].boxes:
                cls_id = int(box.cls)
                cls_name = class_names[cls_id]
                detected_classes.append(cls_name)
                if cls_name == "head":
                    total_persons += 1

                xb1, yb1, xb2, yb2 = box.xyxy[0]
                ix1 = max(int(xb1), x1)
                iy1 = max(int(yb1), y1)
                ix2 = min(int(xb2), x2)
                iy2 = min(int(yb2), y2)
                if ix2 > ix1 and iy2 > iy1:
                    warn_in_area = True

            counter = collections.Counter(detected_classes)
            for c in counter:
                if RISK.get(c, "低危") == "高危":
                    high_risk = True

            stat_info.info(f"👤 实时人数：{total_persons}  |  违规：{dict(counter)}")

            now_ts = time.time()
            now_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            is_alarm = warn_in_area or high_risk

            if is_alarm:
                alert_info.error("⚠️ 检测到违规行为")
                if now_ts - last_save_time > capture_interval:
                    img_path = os.path.join(ALARM_FOLDER, f"{now_str}.jpg")
                    cv2.imwrite(img_path, frame)
                    log_str = f"[{now_str}] 人数={total_persons} | 违规={dict(counter)}\n"
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(log_str)
                    last_save_time = now_ts
            else:
                alert_info.success("✅ 现场安全正常")

            # ====================== 只改这一行！和第一段完全一致 ======================
            frame_rgb = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
            frame_show.image(frame_rgb, use_container_width=True)

        cap.release()
    else:
        alert_info.info("请勾选开启摄像头")

# ====================== 图片检测 ======================
elif task_type == "🖼️ 图片检测":
    conf_threshold = st.slider("检测置信度", 0.1, 0.9, 0.5)
    uploaded = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        res = model.predict(img, conf=conf_threshold, imgsz=640)
        res_img = res[0].plot()
        st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)

# ====================== 视频检测 ======================
elif task_type == "🎞️ 视频检测":
    conf_threshold = st.slider("检测置信度", 0.1, 0.9, 0.5)
    uploaded = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        frame_skip = 2  # 每2帧检测1帧，流畅度翻倍
        count = 0


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % frame_skip != 0:
                # 不检测，直接显示原帧（更流畅）
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                continue

            res = model.predict(frame, conf=conf_threshold, verbose=False, imgsz=480)
            stframe.image(cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
        cap.release()

# ====================== 违规统计管理系统 ======================
elif task_type == "📊 违规统计管理系统":
    st.subheader("📋 违规行为统计报表")

    if st.button("🗑️ 清空所有日志与统计数据", type="primary"):
        if os.path.exists(log_file):
            os.remove(log_file)
        st.success("✅ 所有日志已清空！")
        st.rerun()

    st.markdown("---")

    if not os.path.exists(log_file):
        st.warning("暂无检测记录")
    else:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_count = len(lines)
        st.success(f"📌 总检测记录数：{total_count} 条")

        class_counter = collections.defaultdict(int)
        person_sum = 0

        for line in lines:
            try:
                if "违规={" in line:
                    part = line.split("违规=")[-1].strip()
                    cdict = ast.literal_eval(part)
                    for k, v in cdict.items():
                        class_counter[k] += v
                if "人数=" in line:
                    num = int(line.split("人数=")[1].split(" |")[0])
                    person_sum += num
            except:
                continue

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 👤 累计识别人数")
            st.metric("总人次", person_sum)

        with col2:
            st.markdown("### ⚠️ 违规等级统计")
            for cls, cnt in class_counter.items():
                level = RISK.get(cls, "低危")
                color = COLOR[level]
                st.markdown(f"""<span style='color:{color};font-weight:bold'>• {cls}：{cnt} 次 [{level}]</span>""",
                            unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 违规数据可视化")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.bar(x=list(class_counter.keys()), y=list(class_counter.values())),
                            use_container_width=True)
        with col_b:
            st.plotly_chart(px.pie(names=list(class_counter.keys()), values=list(class_counter.values())),
                            use_container_width=True)

        st.markdown("---")
        st.subheader("📥 导出Excel")
        df = pd.DataFrame({"违规类型": list(class_counter.keys()), "次数": list(class_counter.values())})
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        st.download_button("导出报表", data=output.getvalue(), file_name="report.xlsx")

        st.markdown("---")
        st.subheader("🖼️ 最新告警图片")
        imgs = sorted([f for f in os.listdir(ALARM_FOLDER)], reverse=True)[:6]
        cols = st.columns(3)
        for i, im in enumerate(imgs):
            cols[i % 3].image(os.path.join(ALARM_FOLDER, im))

        st.markdown("---")
        st.subheader("📄 日志")
        st.code("".join(lines[-50:]))