# 🅿️ Parking Lot Analysis System - Usage Guide

## Overview
This system uses **YOLOv8** (`best.pt`) to analyze parking lot videos and provides:
- ✅ Real-time vacancy detection
- 📊 Occupancy statistics
- 🎯 Navigation guidance to nearest free space
- 📍 Detailed space tracking

---

## ⚡ Quick Start

### Option 1: Auto-Detect Video (Recommended)
```bash
# Automatically finds video in Car-Videos/ folder
python parking_integration.py
```

### Option 2: Specify Video File
```bash
# Analyze a specific video
python parking_integration.py --source "Car-Videos/parking_video.mp4"
```

### Option 3: Use Different Model
```bash
# With custom weights
python parking_integration.py --source "Car-Videos/video.mp4" --weights "custom_model.pt"
```

### Option 4: Headless Mode (No Display)
```bash
# Faster processing - doesn't display video
python parking_integration.py --source "Car-Videos/video.mp4" --headless
```

---

## 📊 What You Get

### 1. **Real-Time HUD** (on video)
- **Left Panel**: Total, vacant, and occupied spaces with occupancy bar
- **Right Panel**: Navigation guidance to nearest vacant space

### 2. **Space Status Overlay**
- Shows each parking space status (✓ free / ✗ occupied)
- Color-coded: Green = free, Red = occupied

### 3. **Final Analysis Report** (printed to console)
```
═══════════════════════════════════════════════════════════════════════════════
                    🅿️  PARKING LOT ANALYSIS REPORT 🅿️
═══════════════════════════════════════════════════════════════════════════════

📊 OVERALL STATISTICS:
   Total Spaces        : 24
   ✅ Vacant Spaces    : 8 (33.3%)
   ❌ Occupied Spaces  : 16 (66.7%)
   Occupancy Rate      : 66.7%

📍 VACANT SPACES:
   Section A: A03, A05, A06, A08
   Section B: B02, B04, B07, B11

🚗 OCCUPIED SPACES:
   Section A: A01, A02, A04, A07, A09, A10, A11, A12
   Section B: B01, B03, B05, B06, B08, B09, B10, B12

🎯 NAVIGATION & ASSISTANCE:
   PRIMARY RECOMMENDATION:
   ├─ Target Space    : A03
   ├─ Section         : A
   ├─ Status          : ✅ FREE
   ├─ Walk Time       : ~12s from entry
   └─ Distance        : 245.3px

   NAVIGATION PATH (4 waypoints):
   └─ [START] Entry Gate → (40, 210)
   ├─ [WAYPOINT 1] → (150, 210)
   ├─ [WAYPOINT 2] → (150, 180)
   └─ [DESTINATION] Space A03 at (300, 180) ✓

   ALTERNATIVE SPACES (Same Section A):
   ├─ A05 (~15s walk)
   ├─ A06 (~18s walk)
```

---

## 📁 Project Structure

```
Parking/
├── parking_integration.py      ← Main analysis script (MODIFIED)
├── parking_manager.py          ← Core parking logic & path planning
├── best.pt                      ← YOLOv8 model (4 classes)
├── Car-Videos/                  ← Place your videos here
│   └── parking_video.mp4
├── Processed-Videos/            ← Output videos saved here
└── USAGE_GUIDE.md              ← This file
```

---

## 🎯 How It Works

### Step 1: Load Model
- Loads `best.pt` which detects 4 classes:
  - **Class 0** (Car) → Space OCCUPIED
  - **Class 1** (Space Available) → Space FREE
  - **Class 2** (busy) → Space OCCUPIED
  - **Class 3** (free) → Space FREE

### Step 2: Process Video
- Processes each frame at real-time FPS
- Auto-calibrates space positions from first 5 frames
- Updates space status in real-time

### Step 3: Calculate Statistics
- Counts total, vacant, occupied spaces
- Calculates occupancy percentage
- Groups spaces by section (A, B)

### Step 4: Find Nearest Free Space
- Uses A* pathfinding algorithm
- Calculates shortest path from entry gate
- Estimates walk time to destination

### Step 5: Generate Report
- Prints comprehensive analysis
- Shows navigation guidance
- Lists alternative free spaces

---

## 🚀 Advanced Usage

### In Python Script
```python
from parking_integration import analyze_single_video

# Simple analysis
result = analyze_single_video("Car-Videos/video.mp4")

# With options
result = analyze_single_video(
    "Car-Videos/video.mp4",
    weights="best.pt",
    display=False  # Faster without display
)

# Access results
if result:
    print(f"Free spaces: {result['free']}")
    print(f"Occupancy: {result['occupancy_pct']}%")
```

### As a Jupyter Notebook Cell
```python
from parking_integration import ParkingSystem

system = ParkingSystem(weights="best.pt", video_source="Car-Videos/video.mp4")
system.run(display=True)
```

---

## ⚙️ Configuration

### Model Classes
Edit in `parking_integration.py`:
```python
CLASS_NAMES  = {0: "Car", 1: "Space Available", 2: "busy", 3: "free"}
FREE_CLASSES = {1, 3}      # Classes that mean "FREE"
BUSY_CLASSES = {0, 2}      # Classes that mean "OCCUPIED"
CONFIDENCE_THRESHOLD = 0.45 # Detection confidence cutoff
```

### Parking Lot Layout
Edit in `parking_manager.py`:
```python
layout = ParkingLayout(
    rows=4,        # Number of rows
    cols=6,        # Number of columns per row
    lot_width=800,
    lot_height=600
)
```

---

## 🔧 Troubleshooting

### Video not found
```bash
# Make sure video is in Car-Videos/ folder
# Or use full path:
python parking_integration.py --source "C:/path/to/video.mp4"
```

### Model not found
```bash
# Make sure best.pt is in same directory as script
# Or download from: https://github.com/ultralytics/yolov8
```

### CUDA/GPU issues
```bash
# Run on CPU only (slower but works)
# The model auto-detects GPU, falls back to CPU
```

### Window not displaying on OneDrive path
- The script auto-handles this by saving frames as images
- Check `latest_frame.jpg` for output

---

## 📈 Output Files

### Console Output
- Real-time statistics during processing
- Final analysis report printed to terminal

### Video Output (Optional)
- Can be saved to `Processed-Videos/` folder

### Statistics
- Accessible via the report at end of processing

---

## 💡 Tips

1. **Longer videos = More accurate** - More frames = better calibration
2. **Bright, clear footage = Better detection** - Ensure good lighting
3. **Multiple angles = Comprehensive analysis** - Process multiple videos for full lot coverage
4. **Headless mode faster** - Use `--headless` for faster processing without display

---

## 📞 Support

For issues or questions:
1. Check that `best.pt` exists and is valid
2. Verify video format is supported (.mp4, .avi, .mov, .mkv)
3. Ensure parking_manager.py is in same directory
4. Check Python packages: `pip install ultralytics opencv-python`

---

**Happy Parking! 🅿️**
