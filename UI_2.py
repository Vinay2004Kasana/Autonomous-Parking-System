"""
Autonomous Parking System - Web Interface
Flask backend that serves the UI and runs YOLO detection on uploaded images/videos.
"""

import os
import cv2
import json
import time
import base64
import tempfile
import threading
import numpy as np
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename

# ── Try importing ultralytics; fall back to a mock for demo mode ──────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed – running in DEMO mode.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB
UPLOAD_FOLDER = Path(tempfile.mkdtemp())

# ── Model paths – adjust if your weights live elsewhere ───────────────────────
MODEL_PATHS = [
    "best.pt",
    "runs/detect/train/weights/best.pt",
    "parking_model.pt",
    "yolov8n.pt",  # fallback to COCO model
]

model = None

def load_model():
    global model
    if not YOLO_AVAILABLE:
        return None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model = YOLO(path)
            print(f"[INFO] Loaded model: {path}")
            return model
    # Try downloading YOLOv8n as a demo fallback
    try:
        model = YOLO("yolov8n.pt")
        print("[INFO] Using YOLOv8n (COCO) as demo model.")
    except Exception as e:
        print(f"[WARN] Could not load any model: {e}")
        model = None
    return model

# Class names – update to match YOUR model's class names
# Common label mappings for parking detection models:
OCCUPIED_CLASSES  = {"car", "occupied", "vehicle", "bus", "truck", "motorcycle", "space-occupied"}
EMPTY_CLASSES     = {"empty", "free", "space-empty", "vacant", "available"}

def classify_detections(results):
    """Parse YOLO results into occupied / empty / car counts."""
    occupied = 0
    empty    = 0
    cars     = 0
    boxes    = []

    for r in results:
        for box in r.boxes:
            cls_id   = int(box.cls[0])
            cls_name = r.names[cls_id].lower()
            conf     = float(box.conf[0])
            xyxy     = box.xyxy[0].cpu().numpy().tolist()

            if cls_name in EMPTY_CLASSES:
                empty += 1
                color = (0, 255, 0)   # green
                label_type = "empty"
            elif cls_name in OCCUPIED_CLASSES or cls_name == "car":
                occupied += 1
                if cls_name == "car":
                    cars += 1
                color = (0, 0, 255)   # red
                label_type = "occupied"
            else:
                # Treat anything else as a car / occupied
                occupied += 1
                cars += 1
                color = (0, 120, 255)
                label_type = "car"

            boxes.append({
                "xyxy": xyxy,
                "conf": round(conf, 3),
                "class": cls_name,
                "type": label_type,
                "color": color,
            })

    total = occupied + empty
    pct   = round((occupied / total * 100) if total else 0, 1)
    return {"occupied": occupied, "empty": empty, "cars": cars,
            "total": total, "occupancy_pct": pct, "boxes": boxes}


def draw_boxes(frame, stats):
    """Draw bounding boxes on a frame."""
    overlay = frame.copy()
    for b in stats["boxes"]:
        x1,y1,x2,y2 = map(int, b["xyxy"])
        color = tuple(b["color"])
        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
        label = f"{b['class']} {b['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(overlay, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(overlay, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # HUD overlay
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0,0), (w, 52), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, f"OCCUPIED: {stats['occupied']}",  (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,80,255), 2)
    cv2.putText(frame, f"EMPTY: {stats['empty']}",        (230, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,220,60), 2)
    cv2.putText(frame, f"TOTAL: {stats['total']}  |  {stats['occupancy_pct']}% full",
                (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    return frame


def demo_stats(filename=""):
    """Return plausible fake stats when no model is available."""
    import random, hashlib
    seed = int(hashlib.md5(filename.encode()).hexdigest()[:8], 16)
    rng  = random.Random(seed)
    total    = rng.randint(20, 60)
    occupied = rng.randint(5, total)
    empty    = total - occupied
    return {"occupied": occupied, "empty": empty, "cars": occupied,
            "total": total, "occupancy_pct": round(occupied/total*100,1),
            "boxes": [], "demo": True}


# ────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AutoPark · Detection System</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#06080f;
  --panel:#0d1120;
  --border:#1c2540;
  --accent:#3a7bfd;
  --green:#00e676;
  --red:#ff3b5c;
  --amber:#ffb300;
  --text:#d4deff;
  --muted:#4a5580;
  --glow:rgba(58,123,253,.35);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background:
  radial-gradient(ellipse 60% 40% at 20% 10%,rgba(58,123,253,.08) 0,transparent 60%),
  radial-gradient(ellipse 50% 40% at 80% 80%,rgba(0,230,118,.05) 0,transparent 60%);
  pointer-events:none}

/* NAV */
nav{display:flex;align-items:center;justify-content:space-between;
  padding:14px 32px;border-bottom:1px solid var(--border);
  background:rgba(6,8,15,.8);backdrop-filter:blur(12px);position:sticky;top:0;z-index:100}
.logo{display:flex;align-items:center;gap:10px;font-size:1.35rem;font-weight:700;letter-spacing:.05em}
.logo-icon{width:32px;height:32px;background:var(--accent);border-radius:6px;
  display:flex;align-items:center;justify-content:center;font-size:1rem}
.badge{font-size:.7rem;background:rgba(58,123,253,.2);border:1px solid rgba(58,123,253,.4);
  padding:2px 10px;border-radius:20px;color:var(--accent);letter-spacing:.08em}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* LAYOUT */
.container{max-width:1280px;margin:0 auto;padding:32px 24px}
.hero{text-align:center;padding:40px 0 32px}
.hero h1{font-size:clamp(2rem,4vw,3.2rem);font-weight:700;letter-spacing:.04em;
  background:linear-gradient(135deg,#ffffff 0%,var(--accent) 60%,var(--green) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:var(--muted);margin-top:10px;font-size:1.1rem;letter-spacing:.03em}

/* UPLOAD ZONE */
.upload-zone{border:2px dashed var(--border);border-radius:16px;padding:48px 24px;
  text-align:center;cursor:pointer;transition:.3s;position:relative;overflow:hidden;
  background:rgba(13,17,32,.6)}
.upload-zone:hover,.upload-zone.drag-over{border-color:var(--accent);
  background:rgba(58,123,253,.06);box-shadow:0 0 40px var(--glow)}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-icon{font-size:3rem;margin-bottom:12px;display:block}
.upload-zone h3{font-size:1.3rem;font-weight:600;margin-bottom:6px}
.upload-zone p{color:var(--muted);font-size:.95rem}
.file-types{display:flex;gap:8px;justify-content:center;margin-top:14px;flex-wrap:wrap}
.file-tag{background:rgba(58,123,253,.12);border:1px solid rgba(58,123,253,.3);
  padding:3px 12px;border-radius:20px;font-size:.78rem;color:var(--accent);font-family:'JetBrains Mono'}

/* BUTTON */
.btn{display:inline-flex;align-items:center;gap:8px;padding:12px 28px;border-radius:10px;
  font-family:'Rajdhani',sans-serif;font-size:1rem;font-weight:600;letter-spacing:.06em;
  cursor:pointer;border:none;transition:.25s;text-transform:uppercase}
.btn-primary{background:linear-gradient(135deg,var(--accent),#5a9bff);color:#fff;
  box-shadow:0 4px 24px rgba(58,123,253,.4)}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(58,123,253,.55)}
.btn-primary:disabled{opacity:.4;cursor:not-allowed;transform:none}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--text)}
.btn-outline:hover{border-color:var(--accent);color:var(--accent)}
.btn-row{display:flex;gap:12px;justify-content:center;margin-top:22px;flex-wrap:wrap}

/* CARDS GRID */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:28px 0}
.stat-card{background:var(--panel);border:1px solid var(--border);border-radius:14px;
  padding:22px 20px;position:relative;overflow:hidden;transition:.3s}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.stat-card.occupied::before{background:linear-gradient(90deg,var(--red),#ff7c91)}
.stat-card.empty::before{background:linear-gradient(90deg,var(--green),#69ffb0)}
.stat-card.total::before{background:linear-gradient(90deg,var(--accent),#7ab3ff)}
.stat-card.pct::before{background:linear-gradient(90deg,var(--amber),#ffd754)}
.stat-card .label{font-size:.8rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:10px}
.stat-card .value{font-size:2.8rem;font-weight:700;line-height:1;font-family:'JetBrains Mono'}
.stat-card.occupied .value{color:var(--red)}
.stat-card.empty .value{color:var(--green)}
.stat-card.total .value{color:var(--accent)}
.stat-card.pct .value{color:var(--amber)}
.stat-card .sub{font-size:.82rem;color:var(--muted);margin-top:6px}
.stat-card .icon{position:absolute;right:16px;top:50%;transform:translateY(-50%);
  font-size:2.2rem;opacity:.12}

/* OCCUPANCY BAR */
.occ-bar-wrap{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:22px 24px;margin-bottom:16px}
.occ-bar-label{display:flex;justify-content:space-between;margin-bottom:10px;font-size:.9rem}
.occ-bar-track{height:12px;background:rgba(255,255,255,.07);border-radius:6px;overflow:hidden}
.occ-bar-fill{height:100%;border-radius:6px;transition:width .8s cubic-bezier(.4,0,.2,1);
  background:linear-gradient(90deg,var(--green),var(--amber),var(--red))}

/* RESULT AREA */
.result-section{background:var(--panel);border:1px solid var(--border);border-radius:16px;overflow:hidden;margin-top:24px}
.result-header{padding:16px 24px;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;background:rgba(0,0,0,.2)}
.result-header h3{font-weight:600;font-size:1.1rem;letter-spacing:.05em}
.result-body{padding:0}
.result-img{width:100%;display:block;max-height:600px;object-fit:contain;background:#000}
.video-container{position:relative}
video{width:100%;display:block;background:#000;max-height:600px}

/* HISTORY */
.history-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;margin-top:20px}
.hist-card{background:var(--panel);border:1px solid var(--border);border-radius:12px;
  overflow:hidden;cursor:pointer;transition:.25s}
.hist-card:hover{border-color:var(--accent);transform:translateY(-3px)}
.hist-thumb{width:100%;height:130px;object-fit:cover;background:#0a0f1c;display:block}
.hist-info{padding:12px 14px}
.hist-name{font-size:.9rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hist-stats{display:flex;gap:10px;margin-top:6px;font-size:.8rem;font-family:'JetBrains Mono'}
.hist-occ{color:var(--red)}.hist-emp{color:var(--green)}

/* LOADER */
.loader-overlay{position:fixed;inset:0;background:rgba(6,8,15,.85);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  z-index:999;backdrop-filter:blur(6px);display:none}
.loader-overlay.active{display:flex}
.spinner{width:56px;height:56px;border:3px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;margin-bottom:18px}
@keyframes spin{to{transform:rotate(360deg)}}
.loader-text{font-size:1.1rem;color:var(--text);font-family:'JetBrains Mono';
  letter-spacing:.06em;animation:blink 1.2s steps(2) infinite}
@keyframes blink{50%{opacity:.3}}

/* DEMO BANNER */
.demo-banner{background:rgba(255,179,0,.1);border:1px solid rgba(255,179,0,.3);
  border-radius:10px;padding:12px 18px;margin-bottom:20px;font-size:.9rem;
  color:var(--amber);display:none;align-items:center;gap:10px}
.demo-banner.show{display:flex}

/* SECTION TITLE */
.section-title{font-size:1rem;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);
  margin:32px 0 14px;display:flex;align-items:center;gap:10px}
.section-title::after{content:'';flex:1;height:1px;background:var(--border)}

/* RESPONSIVE */
@media(max-width:600px){nav{padding:12px 16px}.container{padding:20px 14px}
  .stat-card .value{font-size:2.2rem}}

/* TOAST */
.toast{position:fixed;bottom:24px;right:24px;background:var(--panel);
  border:1px solid var(--border);border-radius:10px;padding:14px 20px;
  font-size:.95rem;transform:translateY(80px);opacity:0;transition:.4s;z-index:200;
  max-width:320px;box-shadow:0 8px 32px rgba(0,0,0,.4)}
.toast.show{transform:translateY(0);opacity:1}
.toast.success{border-left:3px solid var(--green)}
.toast.error{border-left:3px solid var(--red)}
</style>
</head>
<body>

<nav>
  <div class="logo">
    <div class="logo-icon">🅿</div>
    AutoPark<span style="color:var(--accent)">·</span>AI
  </div>
  <div class="badge">DETECTION SYSTEM</div>
  <div style="display:flex;align-items:center;gap:8px;font-size:.85rem;color:var(--muted)">
    <div class="status-dot"></div> LIVE
  </div>
</nav>

<div class="loader-overlay" id="loader">
  <div class="spinner"></div>
  <div class="loader-text" id="loaderText">RUNNING DETECTION…</div>
</div>

<div class="toast" id="toast"></div>

<div class="container">
  <div class="hero">
    <h1>Autonomous Parking Detection</h1>
    <p>Upload parking lot images or videos — get instant slot occupancy analysis</p>
  </div>

  <div class="demo-banner" id="demoBanner">
    ⚠️ <strong>Demo Mode:</strong> No YOLO model found — showing simulated results. Place <code>best.pt</code> in the project folder to enable real detection.
  </div>

  <!-- UPLOAD ZONE -->
  <div class="upload-zone" id="uploadZone">
    <input type="file" id="fileInput" accept="image/*,video/*" multiple/>
    <span class="upload-icon">📁</span>
    <h3>Drop files here or click to browse</h3>
    <p>Supports parking lot images and videos</p>
    <div class="file-types">
      <span class="file-tag">.jpg</span><span class="file-tag">.png</span>
      <span class="file-tag">.mp4</span><span class="file-tag">.avi</span>
      <span class="file-tag">.mov</span><span class="file-tag">.mkv</span>
    </div>
  </div>

  <div class="btn-row">
    <button class="btn btn-primary" id="detectBtn" disabled onclick="runDetection()">
      🔍 &nbsp;Run Detection
    </button>
    <button class="btn btn-outline" onclick="clearAll()">✕ &nbsp;Clear</button>
  </div>

  <!-- STATS -->
  <div id="statsSection" style="display:none">
    <div class="section-title">Detection Results</div>

    <div class="stats-grid">
      <div class="stat-card occupied">
        <div class="label">Occupied Slots</div>
        <div class="value" id="valOccupied">—</div>
        <div class="sub" id="subOccupied">Cars detected</div>
        <div class="icon">🚗</div>
      </div>
      <div class="stat-card empty">
        <div class="label">Empty Slots</div>
        <div class="value" id="valEmpty">—</div>
        <div class="sub">Available now</div>
        <div class="icon">🟩</div>
      </div>
      <div class="stat-card total">
        <div class="label">Total Slots</div>
        <div class="value" id="valTotal">—</div>
        <div class="sub">In lot</div>
        <div class="icon">🅿</div>
      </div>
      <div class="stat-card pct">
        <div class="label">Occupancy</div>
        <div class="value" id="valPct">—</div>
        <div class="sub">% of capacity used</div>
        <div class="icon">📊</div>
      </div>
    </div>

    <div class="occ-bar-wrap">
      <div class="occ-bar-label">
        <span style="color:var(--green)">▓ Empty</span>
        <span id="barLabel">—</span>
        <span style="color:var(--red)">Occupied ▓</span>
      </div>
      <div class="occ-bar-track">
        <div class="occ-bar-fill" id="occBar" style="width:0%"></div>
      </div>
    </div>

    <!-- RESULT IMAGE/VIDEO -->
    <div class="result-section" id="resultSection" style="display:none">
      <div class="result-header">
        <h3 id="resultTitle">Detection Output</h3>
        <button class="btn btn-outline" style="padding:6px 14px;font-size:.85rem" onclick="downloadResult()">⬇ Download</button>
      </div>
      <div class="result-body">
        <img id="resultImg" class="result-img" style="display:none"/>
        <div class="video-container" id="videoContainer" style="display:none">
          <video id="resultVideo" controls></video>
        </div>
      </div>
    </div>
  </div>

  <!-- HISTORY -->
  <div id="historySection" style="display:none">
    <div class="section-title">Recent Analyses</div>
    <div class="history-grid" id="historyGrid"></div>
  </div>

</div><!-- /container -->

<script>
const history   = [];
let lastResult  = null;
let selectedFiles = [];

// ── File selection ────────────────────────────────────────────────────────
const fileInput  = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const detectBtn  = document.getElementById('detectBtn');

fileInput.addEventListener('change', e => {
  selectedFiles = Array.from(e.target.files);
  if(selectedFiles.length) onFilesSelected();
});

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  selectedFiles = Array.from(e.dataTransfer.files);
  if(selectedFiles.length) onFilesSelected();
});

function onFilesSelected(){
  const names = selectedFiles.map(f=>f.name).join(', ');
  uploadZone.querySelector('h3').textContent = `${selectedFiles.length} file(s) selected`;
  uploadZone.querySelector('p').textContent = names.length > 60 ? names.slice(0,57)+'…' : names;
  detectBtn.disabled = false;
}

// ── Detection ─────────────────────────────────────────────────────────────
async function runDetection(){
  if(!selectedFiles.length) return;
  showLoader('RUNNING DETECTION…');
  detectBtn.disabled = true;

  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('files', f));

  try{
    const res  = await fetch('/detect', {method:'POST', body:fd});
    const data = await res.json();
    if(data.error){ showToast(data.error,'error'); return; }
    renderStats(data);
    if(data.image_b64) renderImage(data, selectedFiles[0].name);
    else if(data.video_url) renderVideo(data.video_url, selectedFiles[0].name);
    addHistory(data, selectedFiles[0].name);
    if(data.demo) document.getElementById('demoBanner').classList.add('show');
    showToast('Detection complete ✓','success');
  } catch(e){
    showToast('Server error: '+e.message,'error');
  } finally{
    hideLoader();
    detectBtn.disabled = false;
  }
}

function renderStats(d){
  document.getElementById('statsSection').style.display = 'block';
  document.getElementById('valOccupied').textContent = d.occupied;
  document.getElementById('valEmpty').textContent    = d.empty;
  document.getElementById('valTotal').textContent    = d.total;
  document.getElementById('valPct').textContent      = d.occupancy_pct + '%';
  document.getElementById('subOccupied').textContent = `${d.cars || d.occupied} car(s) detected`;
  const pct = d.occupancy_pct;
  document.getElementById('occBar').style.width = pct + '%';
  document.getElementById('barLabel').textContent = `${d.occupied} / ${d.total} slots`;
}

function renderImage(d, fname){
  const img = document.getElementById('resultImg');
  img.src = 'data:image/jpeg;base64,' + d.image_b64;
  img.style.display = 'block';
  document.getElementById('resultVideo').style.display = 'none';
  document.getElementById('videoContainer').style.display = 'none';
  document.getElementById('resultTitle').textContent = fname + ' — Detection Output';
  document.getElementById('resultSection').style.display = 'block';
  lastResult = d;
}

function renderVideo(url, fname){
  const vid = document.getElementById('resultVideo');
  vid.src = url;
  vid.style.display = 'block';
  document.getElementById('resultImg').style.display = 'none';
  document.getElementById('videoContainer').style.display = 'block';
  document.getElementById('resultTitle').textContent = fname + ' — Detection Output';
  document.getElementById('resultSection').style.display = 'block';
}

function addHistory(d, fname){
  history.unshift({name:fname, ...d, ts: new Date().toLocaleTimeString()});
  renderHistory();
}

function renderHistory(){
  if(!history.length) return;
  document.getElementById('historySection').style.display = 'block';
  const grid = document.getElementById('historyGrid');
  grid.innerHTML = history.slice(0,8).map((h,i)=>`
    <div class="hist-card" onclick="restoreHistory(${i})">
      ${h.image_b64
        ? `<img class="hist-thumb" src="data:image/jpeg;base64,${h.image_b64}"/>`
        : `<div class="hist-thumb" style="display:flex;align-items:center;justify-content:center;font-size:2rem">🎬</div>`}
      <div class="hist-info">
        <div class="hist-name">${h.name}</div>
        <div class="hist-stats">
          <span class="hist-occ">🚗 ${h.occupied}</span>
          <span class="hist-emp">🟩 ${h.empty}</span>
          <span style="color:var(--muted)">${h.ts}</span>
        </div>
      </div>
    </div>`).join('');
}

function restoreHistory(i){
  const h = history[i];
  renderStats(h);
  if(h.image_b64) renderImage(h, h.name);
}

function downloadResult(){
  if(!lastResult?.image_b64) return;
  const a = document.createElement('a');
  a.href = 'data:image/jpeg;base64,'+lastResult.image_b64;
  a.download = 'parking_detection.jpg';
  a.click();
}

function clearAll(){
  selectedFiles = [];
  fileInput.value = '';
  uploadZone.querySelector('h3').textContent = 'Drop files here or click to browse';
  uploadZone.querySelector('p').textContent  = 'Supports parking lot images and videos';
  detectBtn.disabled = true;
  document.getElementById('statsSection').style.display   = 'none';
  document.getElementById('resultSection').style.display  = 'none';
  document.getElementById('demoBanner').classList.remove('show');
}

// ── Loader & Toast ────────────────────────────────────────────────────────
function showLoader(msg){
  document.getElementById('loaderText').textContent = msg;
  document.getElementById('loader').classList.add('active');
}
function hideLoader(){ document.getElementById('loader').classList.remove('active'); }

function showToast(msg, type='success'){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast ${type} show`;
  setTimeout(()=>t.classList.remove('show'), 3500);
}
</script>
</body>
</html>"""

# ────────────────────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/detect", methods=["POST"])
def detect():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    file = files[0]  # process first file (extend loop for batch)
    fname = secure_filename(file.filename)
    ext   = Path(fname).suffix.lower()

    save_path = UPLOAD_FOLDER / fname
    file.save(str(save_path))

    is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

    if not YOLO_AVAILABLE or model is None:
        # ── DEMO MODE ──────────────────────────────────────────────────────
        stats = demo_stats(fname)
        if is_video:
            return jsonify({**stats, "video_url": f"/video/{fname}"})
        else:
            # Return original image with no boxes in demo
            with open(str(save_path), "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            return jsonify({**stats, "image_b64": img_b64})

    # ── REAL DETECTION ─────────────────────────────────────────────────────
    if is_video:
        out_path = UPLOAD_FOLDER / f"out_{fname}"
        cap = cv2.VideoCapture(str(save_path))
        fps  = cap.get(cv2.CAP_PROP_FPS) or 25
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        agg = {"occupied":0,"empty":0,"cars":0,"total":0,"occupancy_pct":0.0,"frames":0}
        while True:
            ret, frame = cap.read()
            if not ret: break
            results = model(frame, verbose=False)
            stats   = classify_detections(results)
            frame   = draw_boxes(frame, stats)
            writer.write(frame)
            agg["occupied"] += stats["occupied"]
            agg["empty"]    += stats["empty"]
            agg["cars"]     += stats["cars"]
            agg["frames"]   += 1
        cap.release(); writer.release()

        n = max(agg["frames"], 1)
        occ   = round(agg["occupied"]/n)
        emp   = round(agg["empty"]/n)
        total = occ + emp
        return jsonify({
            "occupied": occ, "empty": emp, "cars": round(agg["cars"]/n),
            "total": total,
            "occupancy_pct": round(occ/total*100, 1) if total else 0,
            "video_url": f"/video/out_{fname}"
        })

    else:
        frame = cv2.imread(str(save_path))
        if frame is None:
            return jsonify({"error": "Could not decode image."}), 400
        results = model(frame, verbose=False)
        stats   = classify_detections(results)
        frame   = draw_boxes(frame, stats)
        _, buf  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img_b64 = base64.b64encode(buf).decode()
        return jsonify({**stats, "image_b64": img_b64})


@app.route("/video/<filename>")
def serve_video(filename):
    path = UPLOAD_FOLDER / secure_filename(filename)
    if not path.exists():
        return "Not found", 404
    return send_file(str(path), mimetype="video/mp4")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AutoPark AI  —  Autonomous Parking Detection System")
    print("=" * 60)
    load_model()
    if not YOLO_AVAILABLE:
        print("  ⚠  ultralytics not found  →  pip install ultralytics")
        print("     Running in DEMO mode (fake counts, original images)")
    print("  🌐  Open:  http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)