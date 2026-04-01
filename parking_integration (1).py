"""
parking_integration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Auto-loads best.pt and maps your 4 classes correctly:
    0 = Car             -> space is OCCUPIED
    1 = Space Available -> space is FREE
    2 = busy            -> space is OCCUPIED
    3 = free            -> space is FREE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import math
import threading
import numpy as np
from pathlib import Path
from parking_manager import ParkingManager

# ── CLASS MAP from your data.yaml ─────────────────────────────────────────────
# names: ['Car', 'Space Available', 'busy', 'free']
CLASS_NAMES  = {0: "Car", 1: "Space Available", 2: "busy", 3: "free"}
FREE_CLASSES = {1, 3}   # 'Space Available' and 'free'
BUSY_CLASSES = {0, 2}   # 'Car' and 'busy'
CONFIDENCE_THRESHOLD = 0.45


# ── AUTO-LOAD best.pt ──────────────────────────────────────────────────────────
def load_model(weights_path="best.pt"):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Run: pip install ultralytics")

    path = Path(weights_path)
    if not path.exists():
        for candidate in ["best.pt",
                          "runs/detect/train/weights/best.pt",
                          "runs/detect/train2/weights/best.pt"]:
            if Path(candidate).exists():
                path = Path(candidate)
                break
        else:
            raise FileNotFoundError(f"Could not find best.pt")

    print(f"[OK] Loaded model: {path}")
    model = YOLO(str(path))
    print(f"[OK] Model classes: {list(model.names.values())}")
    # Verify matches expected
    expected = ['Car', 'Space Available', 'busy', 'free']
    if list(model.names.values()) != expected:
        print(f"[!]  Expected {expected} — update FREE_CLASSES if different")
    return model


# ── SPACE MAPPER ──────────────────────────────────────────────────────────────
class SpaceMapper:
    """Maps YOLO bbox centers to the nearest parking space ID."""

    def __init__(self, manager):
        self.manager = manager
        self.calibrated = False
        self._pixel_map = {}   # space_id -> (camera_px, camera_py)

    def calibrate_from_detections(self, detections, frame_w, frame_h):
        """Auto-learn where spaces appear in the camera frame (first 5 frames)."""
        pts = [(d['cx'], d['cy']) for d in detections
               if d['cls'] in FREE_CLASSES | BUSY_CLASSES]
        if len(pts) < 3:
            return
        # Sort top-to-bottom, left-to-right to match space grid order
        sorted_pts = sorted(pts, key=lambda p: (round(p[1] / 60), p[0]))
        space_ids  = list(self.manager.layout.spaces.keys())
        for i, pt in enumerate(sorted_pts[:len(space_ids)]):
            self._pixel_map[space_ids[i]] = pt
        self.calibrated = True
        print(f"[OK] Calibrated {len(self._pixel_map)} spaces from camera frame")

    def find_space_id(self, cx, cy, frame_w=1920, frame_h=1080):
        """Return nearest space ID for a detected center point."""
        if self._pixel_map:
            best, best_d = None, float("inf")
            for sid, (sx, sy) in self._pixel_map.items():
                d = math.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)
                if d < best_d:
                    best_d, best = d, sid
            return best if best_d < 80 else None
        # Fallback: normalize to lot coordinates
        lot_x = cx / frame_w * self.manager.layout.lot_width
        lot_y = cy / frame_h * self.manager.layout.lot_height
        best, best_d = None, float("inf")
        for space in self.manager.layout.spaces.values():
            d = math.sqrt((space.x - lot_x) ** 2 + (space.y - lot_y) ** 2)
            if d < best_d:
                best_d, best = d, space.id
        return best if best_d < 80 else None


# ── MAIN SYSTEM ───────────────────────────────────────────────────────────────
class ParkingSystem:
    """
    Complete pipeline:
      camera/video -> best.pt -> 4-class detection
      -> space mapping -> counter update -> path planning -> gate display
    """

    def __init__(self, weights="best.pt", video_source=0):
        print("[PARKING SYSTEM] Initializing...")
        self.model   = load_model(weights)
        self.manager = ParkingManager()
        self.mapper  = SpaceMapper(self.manager)
        self.source  = video_source
        self.frame_count = 0
        self.running     = False
        self._lock       = threading.Lock()
        self.latest_stats = {}

        # BGR draw colors per class
        self.colors = {
            0: (50,  100, 255),   # Car           -> blue-red
            1: (0,   255, 150),   # Space Avail   -> cyan-green
            2: (30,  30,  220),   # busy          -> red
            3: (0,   230, 80),    # free          -> green
        }

    # ── PROCESS ONE FRAME ─────────────────────────────────────────────────────
    def process_frame(self, frame):
        """
        Pass one BGR frame through best.pt.
        Returns annotated frame + live stats dict.

        HOW THE 4 CLASSES ARE USED:
          Class 0 (Car)            -> marks space OCCUPIED
          Class 1 (Space Avail)    -> marks space FREE
          Class 2 (busy)           -> marks space OCCUPIED
          Class 3 (free)           -> marks space FREE
        Only classes 1/2/3 are used for space status.
        Class 0 (Car) is used as a secondary confirmation of occupancy.
        """
        self.frame_count += 1
        h, w = frame.shape[:2]

        results    = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append({
                "cls": cls, "conf": conf,
                "cx": cx,   "cy": cy,
                "bbox": (x1, y1, x2, y2),
                "label": CLASS_NAMES.get(cls, str(cls)),
                "is_free": cls in FREE_CLASSES,
            })

        # Auto-calibrate first 5 frames
        if not self.mapper.calibrated and self.frame_count <= 5:
            self.mapper.calibrate_from_detections(detections, w, h)

        # Feed into parking manager (real-time counter update)
        yolo_updates = []
        for d in detections:
            # Use space-specific classes (1=avail, 2=busy, 3=free)
            # For class 0 (Car) we also mark as occupied
            if d['cls'] in {0, 1, 2, 3}:
                space_id = self.mapper.find_space_id(d['cx'], d['cy'], w, h)
                if space_id:
                    yolo_updates.append({
                        "space_id": space_id,
                        "is_free":  d['is_free'],
                        "confidence": d['conf'],
                    })

        self.manager.update_from_yolo(yolo_updates)

        # Draw and return
        annotated = self._draw_detections(frame.copy(), detections)
        annotated = self._draw_hud(annotated)
        stats     = self.manager.get_stats()
        with self._lock:
            self.latest_stats = stats
        return annotated, stats

    # ── DRAWING ───────────────────────────────────────────────────────────────
    def _draw_detections(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            color = self.colors.get(d['cls'], (200, 200, 200))
            label = f"{d['label']} {d['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        return frame

    def _draw_hud(self, frame):
        """Real-time counter HUD drawn directly on the video frame."""
        stats = self.manager.get_stats()
        nav   = self.manager.get_navigation_info()
        rec   = nav.get("recommended_space", "FULL")

        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (300, 170), (10, 18, 28), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (8, 8), (300, 170), (0, 200, 120), 1)

        f = cv2.FONT_HERSHEY_SIMPLEX
        y = 35
        cv2.putText(frame, "PARKVISION  best.pt", (18, y), f, 0.55, (0, 220, 160), 2); y += 28
        cv2.putText(frame, f"Total  : {stats['total']}", (18, y), f, 0.5, (200, 220, 255), 1); y += 24
        cv2.putText(frame, f"Free   : {stats['free']}",  (18, y), f, 0.5, (0, 255, 150), 2); y += 24
        cv2.putText(frame, f"Busy   : {stats['busy']}",  (18, y), f, 0.5, (50, 80, 255), 2);  y += 24
        cv2.putText(frame, f"Go to  : Space {rec}",      (18, y), f, 0.5, (0, 200, 255), 2)

        # Occupancy progress bar
        bx, by, bw, bh = 18, 158, 270, 8
        filled = int(bw * stats['busy'] / max(stats['total'], 1))
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 60), -1)
        bar_c = (0, 255, 100) if stats['free'] > 5 else (0, 160, 255) if stats['free'] > 2 else (0, 50, 255)
        cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), bar_c, -1)

        return frame

    # ── GATE TRIGGER ─────────────────────────────────────────────────────────
    def on_vehicle_at_gate(self, prefer_section=None):
        """
        Call when a vehicle arrives at the gate.
        Prints info and returns navigation dict.
        """
        space_id = self.manager.find_nearest_free_space(prefer_section)
        nav   = self.manager.get_navigation_info(space_id)
        stats = self.manager.get_stats()
        print("\n" + "=" * 50)
        print("  VEHICLE AT GATE")
        print(f"  Free  : {stats['free']} / {stats['total']}")
        print(f"  Busy  : {stats['busy']}")
        print(f"  Go to : Space {nav.get('recommended_space')} (Section {nav.get('section')})")
        print(f"  Walk  : ~{nav.get('estimated_walk_seconds')}s")
        print("=" * 50 + "\n")
        return nav

    # ── COUNTER UPDATE (gate sensor hook) ─────────────────────────────────────
    def on_vehicle_entered(self, space_id):
        self.manager.update_space(space_id, is_free=False)
        print(f"[+] Space {space_id} OCCUPIED — Free: {self.manager.get_stats()['free']}")

    def on_vehicle_left(self, space_id):
        self.manager.update_space(space_id, is_free=True)
        print(f"[-] Space {space_id} FREED   — Free: {self.manager.get_stats()['free']}")

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────
    def run(self, display=True, gate_sim_every=90):
        """
        Run the full pipeline.

        Usage:
            # Webcam
            system = ParkingSystem("best.pt", video_source=0)
            system.run()

            # Video file from Car-Videos/
            system = ParkingSystem("best.pt", video_source="Car-Videos/video.mp4")
            system.run()
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"[OK] Video: {int(cap.get(3))}x{int(cap.get(4))} @ {fps:.0f}fps")
        print("[OK] Running best.pt — press Q to quit\n")

        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
                continue

            annotated, stats = self.process_frame(frame)

            if gate_sim_every and self.frame_count % gate_sim_every == 0:
                self.on_vehicle_at_gate()

            if display:
                cv2.imshow("ParkVision", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.running = False
        cap.release()
        if display:
            cv2.destroyAllWindows()


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--source",  default="0",   help="0=webcam or path/to/video.mp4")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    ParkingSystem(weights=args.weights, video_source=src).run(display=not args.headless)
