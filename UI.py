import io, os, base64, tempfile, time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoPark AI",
    page_icon="🅿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #06080f !important;
    color: #d4deff !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
        radial-gradient(ellipse 60% 40% at 15% 10%, rgba(58,123,253,.10) 0, transparent 60%),
        radial-gradient(ellipse 50% 40% at 85% 85%, rgba(0,230,118,.06) 0, transparent 60%);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid #1c2540 !important;
}
[data-testid="stSidebar"] * { color: #d4deff !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0d1120;
    border: 1px solid #1c2540;
    border-radius: 14px;
    padding: 1.1rem 1.2rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #3a7bfd, #7ab3ff);
}
[data-testid="stMetricLabel"] { color: #4a5580 !important; font-family: 'Rajdhani' !important; font-size: .8rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'JetBrains Mono' !important; font-size: 2.2rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono' !important; font-size: .8rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3a7bfd, #5a9bff) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    letter-spacing: .08em !important; text-transform: uppercase !important;
    padding: .55rem 1.6rem !important;
    box-shadow: 0 4px 20px rgba(58,123,253,.35) !important;
    transition: .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(58,123,253,.55) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(13,17,32,.7) !important;
    border: 2px dashed #1c2540 !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    transition: .3s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #3a7bfd !important; }

/* ── Progress / info boxes ── */
.stAlert { border-radius: 10px !important; }
[data-testid="stInfo"] { background: rgba(58,123,253,.12) !important; border: 1px solid rgba(58,123,253,.3) !important; color: #7ab3ff !important; }
[data-testid="stSuccess"] { background: rgba(0,230,118,.10) !important; border: 1px solid rgba(0,230,118,.3) !important; color: #00e676 !important; }
[data-testid="stWarning"] { background: rgba(255,179,0,.10) !important; border: 1px solid rgba(255,179,0,.3) !important; color: #ffb300 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0d1120 !important; border: 1px solid #1c2540 !important; border-radius: 12px !important;
}
[data-testid="stExpanderToggleIcon"] { color: #3a7bfd !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: #0d1120 !important; border-bottom: 1px solid #1c2540 !important; }
[data-baseweb="tab"] { color: #4a5580 !important; font-family: 'Rajdhani' !important; font-weight: 600 !important; letter-spacing: .06em !important; }
[aria-selected="true"][data-baseweb="tab"] { color: #3a7bfd !important; border-bottom-color: #3a7bfd !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #3a7bfd !important; }

/* ── Stat banner (custom HTML) ── */
.stat-banner {
    display: flex; align-items: stretch; gap: 0;
    border: 1px solid #1c2540; border-radius: 14px; overflow: hidden;
    margin: 1rem 0;
}
.stat-cell {
    flex: 1; padding: 1.2rem 1.4rem;
    border-right: 1px solid #1c2540;
    background: #0d1120;
}
.stat-cell:last-child { border-right: none; }
.stat-cell .lbl { font-size: .75rem; letter-spacing: .12em; color: #4a5580; text-transform: uppercase; margin-bottom: .4rem; }
.stat-cell .val { font-family: 'JetBrains Mono'; font-size: 2.4rem; font-weight: 700; line-height: 1; }
.stat-cell .sub { font-size: .8rem; color: #4a5580; margin-top: .3rem; }
.occ  .val { color: #ff3b5c; }
.emp  .val { color: #00e676; }
.tot  .val { color: #3a7bfd; }
.pct  .val { color: #ffb300; }

/* ── Section headings ── */
.sec-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: .85rem; text-transform: uppercase;
    letter-spacing: .16em; color: #4a5580;
    margin: 1.6rem 0 .7rem;
    display: flex; align-items: center; gap: .7rem;
}
.sec-title::after { content: ''; flex: 1; height: 1px; background: #1c2540; }

/* ── Logo / header ── */
.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: .4rem 0 1.2rem;
}
.logo-box {
    width: 46px; height: 46px; background: #3a7bfd; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; font-weight: 900; color: #fff; flex-shrink: 0;
}
.logo-text h1 { font-size: 1.7rem; font-weight: 700; letter-spacing: .04em; margin: 0; line-height: 1; }
.logo-text p  { color: #4a5580; font-size: .85rem; letter-spacing: .06em; margin: 0; }

/* ── History card ── */
.hist-row { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 6px; }
.hist-card {
    flex: 0 0 160px; background: #0d1120; border: 1px solid #1c2540;
    border-radius: 10px; overflow: hidden; cursor: pointer;
}
.hist-card img { width: 100%; height: 90px; object-fit: cover; display: block; background: #0a0f1c; }
.hist-card .hinfo { padding: 8px 10px; }
.hist-card .hname { font-size: .8rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-card .hstats { display: flex; gap: 8px; margin-top: 4px; font-family: 'JetBrains Mono'; font-size: .72rem; }
.hocc { color: #ff3b5c; } .hemp { color: #00e676; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATHS = [
    "best.pt",
    "runs/detect/train/weights/best.pt",
    "parking_model.pt",
    "yolov8n.pt",
]
MODEL_PATHS = DEFAULT_MODEL_PATHS.copy()

OCCUPIED_CLASSES = {"car", "busy"}
EMPTY_CLASSES    = {"space available", "free"}

@st.cache_resource(show_spinner=False)
def load_model():
    if not YOLO_AVAILABLE:
        return None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return YOLO(p)
    try:
        return YOLO("yolov8n.pt")   # auto-download fallback
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def classify_detections(results, conf_thresh=0.35):
    busy = free = cars = 0
    boxes = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            cls_id   = int(box.cls[0])
            cls_name = r.names[cls_id].lower()
            xyxy     = box.xyxy[0].cpu().numpy().tolist()
            if cls_name in EMPTY_CLASSES:
                free += 1
                color, ltype = (0, 230, 118), "free"
            elif cls_name in OCCUPIED_CLASSES:
                busy += 1
                if cls_name == "car": cars += 1
                color, ltype = (255, 59, 92), "busy"
            else:
                busy += 1; cars += 1
                color, ltype = (255, 140, 0), "car"
            boxes.append({"xyxy": xyxy, "conf": conf, "class": cls_name, "type": ltype, "color": color})
    total = busy + free
    return {
        "busy": busy, "free": free, "cars": cars,
        "total": total,
        "occupancy_pct": round(busy / total * 100, 1) if total else 0,
        "boxes": boxes,
    }


def draw_boxes(frame, stats):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for b in stats["boxes"]:
        x1, y1, x2, y2 = map(int, b["xyxy"])
        c = b["color"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, 2)
        label = f"{b['class']} {b['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), c, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.rectangle(overlay, (0, 0), (w, 48), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.putText(frame, f"BUSY: {stats['busy']}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 80, 100), 2)
    cv2.putText(frame, f"FREE: {stats['free']}", (230, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 220, 100), 2)
    cv2.putText(frame,
                f"TOTAL: {stats['total']}  |  {stats['occupancy_pct']}% occupied",
                (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 170, 210), 1)
    return frame


def demo_stats(seed_str=""):
    import random, hashlib
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng  = random.Random(seed)
    total    = rng.randint(20, 60)
    busy = rng.randint(5, total)
    free    = total - busy
    return {"busy": busy, "free": free, "cars": busy,
            "total": total, "occupancy_pct": round(busy / total * 100, 1),
            "boxes": [], "demo": True}


def process_image(img_array, model, conf_thresh=0.35):
    if model is None or not YOLO_AVAILABLE:
        return img_array, demo_stats("img")
    results = model(img_array, verbose=False, conf=conf_thresh)
    stats   = classify_detections(results, conf_thresh)
    out     = draw_boxes(img_array.copy(), stats)
    return out, stats


def process_video(video_bytes, fname, model, conf_thresh=0.35, sample_rate=1):
    tmp_in  = tempfile.NamedTemporaryFile(suffix=Path(fname).suffix, delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_in.write(video_bytes); tmp_in.close(); tmp_out.close()

    cap    = cv2.VideoCapture(tmp_in.name)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out.name, fourcc, fps, (W, H))

    agg      = {"busy": 0, "free": 0, "cars": 0, "frames": 0}
    per_frame = []
    frame_idx = 0

    prog_bar  = st.progress(0, text="Processing video…")
    stat_slot = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % max(1, sample_rate) != 0:
            writer.write(frame)
            continue

        if model and YOLO_AVAILABLE:
            results = model(frame, verbose=False, conf=conf_thresh)
            stats   = classify_detections(results, conf_thresh)
            frame   = draw_boxes(frame.copy(), stats)
        else:
            stats = demo_stats(f"frame_{frame_idx}")
            stats["demo"] = True

        writer.write(frame)
        agg["busy"] += stats["busy"]
        agg["free"]    += stats["free"]
        agg["cars"]     += stats["cars"]
        agg["frames"]   += 1
        per_frame.append({"frame": frame_idx,
                           "busy": stats["busy"],
                           "free": stats["free"]})

        pct = min(frame_idx / max(total_frames, 1), 1.0)
        prog_bar.progress(pct, text=f"Processing frame {frame_idx}/{total_frames}…")
        if agg["frames"] % 10 == 0:
            stat_slot.info(f"🚗 Busy avg: **{agg['busy']//agg['frames']}**  "
                           f"  🟩 Free avg: **{agg['free']//agg['frames']}**")

    cap.release(); writer.release()
    prog_bar.empty(); stat_slot.empty()
    os.unlink(tmp_in.name)

    n = max(agg["frames"], 1)
    occ   = round(agg["busy"] / n)
    emp   = round(agg["free"]    / n)
    total = occ + emp
    summary = {
        "busy": occ, "free": emp, "cars": round(agg["cars"] / n),
        "total": total,
        "occupancy_pct": round(occ / total * 100, 1) if total else 0,
        "per_frame": per_frame,
    }
    return tmp_out.name, summary


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY GAUGE
# ─────────────────────────────────────────────────────────────────────────────
def make_gauge(pct):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 34, "family": "JetBrains Mono", "color": "#d4deff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5580",
                     "tickfont": {"color": "#4a5580", "size": 11}},
            "bar": {"color": "#3a7bfd"},
            "bgcolor": "#0d1120",
            "bordercolor": "#1c2540",
            "steps": [
                {"range": [0,  40], "color": "rgba(0,230,118,.18)"},
                {"range": [40, 70], "color": "rgba(255,179,0,.15)"},
                {"range": [70,100], "color": "rgba(255,59,92,.18)"},
            ],
            "threshold": {"line": {"color": "#ff3b5c", "width": 3}, "value": pct},
        },
        title={"text": "Occupancy Rate", "font": {"size": 14, "color": "#4a5580", "family": "Rajdhani"}},
    ))
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="#0d1120", font_color="#d4deff",
    )
    return fig


def make_timeline(per_frame):
    frames = [p["frame"]    for p in per_frame]
    occ    = [p["busy"] for p in per_frame]
    emp    = [p["free"]    for p in per_frame]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames, y=occ, name="Busy",
                             line=dict(color="#ff3b5c", width=2),
                             fill="tozeroy", fillcolor="rgba(255,59,92,.12)"))
    fig.add_trace(go.Scatter(x=frames, y=emp, name="Free",
                             line=dict(color="#00e676", width=2),
                             fill="tozeroy", fillcolor="rgba(0,230,118,.10)"))
    fig.update_layout(
        height=260, paper_bgcolor="#0d1120", plot_bgcolor="#06080f",
        font=dict(family="Rajdhani", color="#d4deff"),
        xaxis=dict(title="Frame", gridcolor="#1c2540", color="#4a5580"),
        yaxis=dict(title="Count",  gridcolor="#1c2540", color="#4a5580"),
        legend=dict(bgcolor="#0d1120", bordercolor="#1c2540", borderwidth=1),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:.8rem 0 1.2rem'>
      <div style='font-size:2rem'>🅿</div>
      <div style='font-family:Rajdhani;font-size:1.25rem;font-weight:700;letter-spacing:.06em'>AutoPark AI</div>
      <div style='color:#4a5580;font-size:.8rem;letter-spacing:.1em'>DETECTION SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**⚙️ Detection Settings**")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05,
                            help="Minimum confidence for a detection to be counted")
    sample_rate = st.slider("Video Sample Rate", 1, 10, 1, 1,
                            help="Process every Nth frame (higher = faster but less accurate)")

    st.divider()
    st.markdown("**🏷 Model Info**")
    custom_model_paths = st.text_input(
        "Custom Model Paths (comma-separated)",
        value=",".join(DEFAULT_MODEL_PATHS),
        help="Specify model file paths separated by commas, or leave as default."
    )
    MODEL_PATHS = [p.strip() for p in custom_model_paths.split(",") if p.strip()]
    model = load_model()
    if not YOLO_AVAILABLE:
        st.warning("ultralytics not installed\n\n`pip install ultralytics`")
    elif model is None:
        st.warning("No model found.\n\nPlace `best.pt` in the project folder or specify a custom path above.")
    else:
        st.success("Model loaded ✓")
        for p in MODEL_PATHS:
            if os.path.exists(p):
                st.caption(f"📦 `{p}`")
                break

    st.divider()
    st.markdown("**🎨 Class Mapping**")
    with st.expander("Occupied classes"):
        st.code(", ".join(sorted(OCCUPIED_CLASSES)))
    with st.expander("Empty classes"):
        st.code(", ".join(sorted(EMPTY_CLASSES)))

    if st.session_state.history:
        st.divider()
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="logo-box">🅿</div>
  <div class="logo-text">
    <h1>AutoPark <span style='color:#3a7bfd'>AI</span></h1>
    <p>AUTONOMOUS PARKING DETECTION SYSTEM</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not YOLO_AVAILABLE or model is None:
    st.warning("⚠️ **Demo Mode** — No YOLO model detected. Displaying simulated results. "
               "Place `best.pt` in the project root to enable real detection.")

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_detect, tab_history, tab_about = st.tabs(["🔍  Detect", "📋  History", "ℹ️  About"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DETECT
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:

    uploaded = st.file_uploader(
        "Upload a parking lot image or video",
        type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv", "m4v"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0;color:#4a5580'>
          <div style='font-size:3rem;margin-bottom:1rem'>📁</div>
          <div style='font-size:1.1rem;font-family:Rajdhani;letter-spacing:.06em'>
            Drop a parking lot image or video above to begin
          </div>
          <div style='font-size:.85rem;margin-top:.5rem'>
            Supported: JPG · PNG · MP4 · AVI · MOV · MKV
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        fname = uploaded.name
        ext   = Path(fname).suffix.lower()
        is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown('<div class="sec-title">Input Preview</div>', unsafe_allow_html=True)
            if is_video:
                st.video(uploaded)
            else:
                pil_img = Image.open(uploaded).convert("RGB")
                st.image(pil_img, use_container_width=True, caption=fname)

        with col_right:
            st.markdown('<div class="sec-title">Run Detection</div>', unsafe_allow_html=True)

            run_btn = st.button("🔍  Run Detection", use_container_width=True)

            if run_btn:
                uploaded.seek(0)
                raw = uploaded.read()

                if is_video:
                    # ── VIDEO ────────────────────────────────────────────
                    with st.spinner(""):
                        out_path, stats = process_video(
                            raw, fname, model, conf_thresh, sample_rate)

                    # Stats banner
                    st.markdown(f"""
                    <div class="stat-banner">
                      <div class="stat-cell occ">
                        <div class="lbl">Busy</div>
                        <div class="val">{stats['busy']}</div>
                        <div class="sub">avg / frame</div>
                      </div>
                      <div class="stat-cell emp">
                        <div class="lbl">Free</div>
                        <div class="val">{stats['free']}</div>
                        <div class="sub">avg / frame</div>
                      </div>
                      <div class="stat-cell tot">
                        <div class="lbl">Total</div>
                        <div class="val">{stats['total']}</div>
                        <div class="sub">slots detected</div>
                      </div>
                      <div class="stat-cell pct">
                        <div class="lbl">Occupancy</div>
                        <div class="val">{stats['occupancy_pct']}%</div>
                        <div class="sub">avg utilisation</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if PLOTLY_AVAILABLE:
                        g_col, t_col = st.columns([1, 1.4])
                        with g_col:
                            st.plotly_chart(make_gauge(stats["occupancy_pct"]),
                                            use_container_width=True)
                        with t_col:
                            if stats.get("per_frame"):
                                st.markdown('<div class="sec-title">Frame Timeline</div>',
                                            unsafe_allow_html=True)
                                st.plotly_chart(make_timeline(stats["per_frame"]),
                                                use_container_width=True)

                    # Annotated video download
                    with open(out_path, "rb") as f:
                        st.download_button("⬇ Download Annotated Video", f,
                                           file_name=f"detected_{fname}",
                                           mime="video/mp4", use_container_width=True)
                    os.unlink(out_path)

                    # Add to history
                    st.session_state.history.insert(0, {
                        "name": fname, "type": "video",
                        "thumb": None, **stats
                    })

                else:
                    # ── IMAGE ─────────────────────────────────────────────
                    img_arr = np.array(pil_img)
                    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

                    with st.spinner("Running YOLO detection…"):
                        out_bgr, stats = process_image(img_bgr, model, conf_thresh)

                    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                    out_pil = Image.fromarray(out_rgb)

                    st.markdown('<div class="sec-title">Annotated Output</div>',
                                unsafe_allow_html=True)
                    st.image(out_pil, use_container_width=True)

                    # Stats banner
                    st.markdown(f"""
                    <div class="stat-banner">
                      <div class="stat-cell occ">
                        <div class="lbl">Occupied</div>
                        <div class="val">{stats['occupied']}</div>
                        <div class="sub">{stats.get('cars', stats['occupied'])} car(s)</div>
                      </div>
                      <div class="stat-cell emp">
                        <div class="lbl">Empty</div>
                        <div class="val">{stats['empty']}</div>
                        <div class="sub">free slots</div>
                      </div>
                      <div class="stat-cell tot">
                        <div class="lbl">Total</div>
                        <div class="val">{stats['total']}</div>
                        <div class="sub">in lot</div>
                      </div>
                      <div class="stat-cell pct">
                        <div class="lbl">Occupancy</div>
                        <div class="val">{stats['occupancy_pct']}%</div>
                        <div class="sub">capacity used</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if PLOTLY_AVAILABLE:
                        st.plotly_chart(make_gauge(stats["occupancy_pct"]),
                                        use_container_width=True)

                    # Download annotated image
                    buf = io.BytesIO()
                    out_pil.save(buf, format="JPEG", quality=90)
                    st.download_button("⬇ Download Annotated Image", buf.getvalue(),
                                       file_name=f"detected_{fname}",
                                       mime="image/jpeg", use_container_width=True)

                    # Thumb for history
                    thumb_buf = io.BytesIO()
                    out_pil.resize((200, 130)).save(thumb_buf, format="JPEG", quality=70)
                    thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode()

                    st.session_state.history.insert(0, {
                        "name": fname, "type": "image",
                        "thumb": thumb_b64, **stats
                    })

                st.success("✓ Detection complete")

            else:
                st.info("👆 Press **Run Detection** to analyse this file.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    if not st.session_state.history:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0;color:#4a5580'>
          <div style='font-size:2.5rem;margin-bottom:.8rem'>📋</div>
          <div style='font-family:Rajdhani;letter-spacing:.06em'>No analyses yet</div>
          <div style='font-size:.85rem;margin-top:.4rem'>Run detection on a file to see results here</div>
        </div>""", unsafe_allow_html=True)
    else:
        for i, h in enumerate(st.session_state.history[:10]):
            with st.expander(f"{'🖼' if h['type']=='image' else '🎬'}  {h['name']}", expanded=(i==0)):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Busy", h["busy"], delta=None)
                c2.metric("Free",    h["free"],    delta=None)
                c3.metric("Total",    h["total"],    delta=None)
                c4.metric("Occupancy", f"{h['occupancy_pct']}%", delta=None)
                if h.get("thumb"):
                    img_data = base64.b64decode(h["thumb"])
                    st.image(Image.open(io.BytesIO(img_data)),
                             width=220, caption="Annotated thumbnail")
                if h.get("per_frame") and PLOTLY_AVAILABLE:
                    st.plotly_chart(make_timeline(h["per_frame"]),
                                    use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
<div style='max-width:680px;margin:0 auto'>

<div class="sec-title">About This App</div>

<p style='color:#8090c0;line-height:1.7'>
AutoPark AI is the web interface for the <strong style='color:#d4deff'>Autonomous Parking System</strong> project.
It runs a YOLOv8 detection model on parking lot images and videos to count
<span style='color:#ff3b5c'>occupied</span> and <span style='color:#00e676'>empty</span>
slots in real time.
</p>

<div class="sec-title">Setup</div>

```bash
# Install dependencies
pip install streamlit ultralytics opencv-python numpy pillow plotly

# Place your trained model in the project root
#   best.pt  OR  parking_model.pt

# Launch
streamlit run streamlit_parking_app.py
```

<div class="sec-title">Customise Class Names</div>

Edit these two sets at the top of the script to match your model's labels:

```python
OCCUPIED_CLASSES = {"car", "occupied", "vehicle", "space-occupied", ...}
EMPTY_CLASSES    = {"empty", "free", "space-empty", "vacant", ...}
```

<div class="sec-title">Features</div>

| Feature | Detail |
|---|---|
| Image detection | JPG · PNG · BMP · WEBP |
| Video detection | MP4 · AVI · MOV · MKV |
| Confidence slider | Tune detection sensitivity |
| Video sample rate | Skip frames for faster processing |
| Occupancy gauge | Plotly gauge chart |
| Frame timeline | Occupied vs empty over time |
| Download | Save annotated image or video |
| History | Last 10 analyses with thumbnails |
| Demo mode | Works without a model |

<br>
<div style='color:#4a5580;font-size:.8rem;letter-spacing:.08em'>
  AUTOPARK AI · AUTONOMOUS PARKING DETECTION · YOLOv8
</div>

</div>
    """, unsafe_allow_html=True)