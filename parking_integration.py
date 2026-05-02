"""
parking_integration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEATURES:
  • Processes videos ONE BY ONE (pass a folder or list of files)
  • Detects 4 classes:
      0 = Car             -> space is OCCUPIED
      1 = Space Available -> space is FREE
      2 = busy            -> space is OCCUPIED
      3 = free            -> space is FREE
  • Draws SPACE NUMBERS clearly on the video frame
  • Draws A* PATH on the video frame for driver assistance
  • HUD shows VACANT / FULL counters prominently

Usage:
    # Single video
    python parking_integration.py --source video.mp4

    # All videos in a folder (processed one by one)
    python parking_integration.py --folder Car-Videos/

    # Multiple specific files
    python parking_integration.py --source v1.mp4 v2.mp4 v3.mp4

    # Headless (no display window)
    python parking_integration.py --folder Car-Videos/ --headless
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import math
import json
import threading
import numpy as np
import argparse
import sys
import time
from pathlib import Path
from parking_manager import ParkingManager

# Optional: for sending detections to Flask API
try:
    import requests as _requests
except ImportError:
    _requests = None

# ── CLASS MAP from your data.yaml ─────────────────────────────────────────────
CLASS_NAMES  = {0: "Car", 1: "Space Available", 2: "busy", 3: "free"}
FREE_CLASSES = {1, 3}
BUSY_CLASSES = {0, 2}
CONFIDENCE_THRESHOLD = 0.30

# ── COLORS (BGR) ──────────────────────────────────────────────────────────────
COL_FREE     = (0,   230, 80)
COL_BUSY     = (40,  40,  220)
COL_PATH     = (255, 200, 0)
COL_HUD_BG   = (12,  20,  32)
COL_PANEL_BG = (18,  28,  45)
COL_WHITE    = (255, 255, 255)

FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTM = cv2.FONT_HERSHEY_SIMPLEX


# ── AUTO-LOAD best.pt ─────────────────────────────────────────────────────────
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
            raise FileNotFoundError("Could not find best.pt")

    print(f"[OK] Loaded model: {path}")
    model = YOLO(str(path))
    print(f"[OK] Model classes: {list(model.names.values())}")
    return model


# ── SPACE MAPPER ──────────────────────────────────────────────────────────────
class SpaceMapper:
    """Maps YOLO bbox centers to the nearest parking space ID."""

    def __init__(self, manager):
        self.manager = manager
        self.calibrated = False
        self._pixel_map = {}   # space_id -> (camera_px, camera_py)

    def calibrate_from_detections(self, detections, frame_w, frame_h):
        pts = [(d['cx'], d['cy']) for d in detections
               if d['cls'] in FREE_CLASSES | BUSY_CLASSES]
        if len(pts) < 3:
            return
            
        total_detected = len(pts)
        
        # Initialize dynamic layout locally
        self.manager.init_dynamic_layout(total_detected)
        
        # Initialize dynamic layout remotely if API is connected
        if getattr(self.manager, 'api_url', None) and _requests:
            try:
                _requests.post(
                    f"{self.manager.api_url}/init_spaces",
                    json={"total": total_detected},
                    timeout=2
                )
            except Exception:
                pass

        sorted_pts = sorted(pts, key=lambda p: (round(p[1] / 60), p[0]))
        space_ids  = list(self.manager.layout.spaces.keys())
        for i, pt in enumerate(sorted_pts[:len(space_ids)]):
            self._pixel_map[space_ids[i]] = pt
        self.calibrated = True
        print(f"[OK] Calibrated {len(self._pixel_map)} spaces dynamically from frame")

    def find_space_id(self, cx, cy, frame_w=1920, frame_h=1080):
        if self._pixel_map:
            best, best_d = None, float("inf")
            for sid, (sx, sy) in self._pixel_map.items():
                d = math.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)
                if d < best_d:
                    best_d, best = d, sid
            return best if best_d < 80 else None
        lot_x = cx / frame_w * self.manager.layout.lot_width
        lot_y = cy / frame_h * self.manager.layout.lot_height
        best, best_d = None, float("inf")
        for space in self.manager.layout.spaces.values():
            d = math.sqrt((space.x - lot_x) ** 2 + (space.y - lot_y) ** 2)
            if d < best_d:
                best_d, best = d, space.id
        return best if best_d < 80 else None

    def get_space_pixel(self, space_id):
        """Return camera-frame pixel coords for a space (for path overlay)."""
        return self._pixel_map.get(space_id)


# ── A* PATH OVERLAY ───────────────────────────────────────────────────────────
class PathOverlay:
    """
    Draws the A* path (from parking_manager) onto the actual video frame.
    The manager uses 'lot coordinates' (800×600 virtual pixels).
    We scale those to the real camera frame size.
    """

    def __init__(self, manager, mapper):
        self.manager = manager
        self.mapper  = mapper
        self._anim_offset = 0

    def _lot_to_frame(self, lot_x, lot_y, fw, fh):
        lw = self.manager.layout.lot_width
        lh = self.manager.layout.lot_height
        fx = int(lot_x / lw * fw)
        fy = int(lot_y / lh * fh)
        return fx, fy

    def draw(self, frame, nav_info):
        """Draw animated A* path and destination marker on frame."""
        path = nav_info.get("path", [])
        rec  = nav_info.get("recommended_space")
        if not path or len(path) < 2:
            return frame

        h, w = frame.shape[:2]
        self._anim_offset = (self._anim_offset + 2) % 40

        # Convert lot-space path points → frame pixels
        pts = []
        for i, p in enumerate(path):
            if i == len(path) - 1 and rec:
                pix = self.mapper.get_space_pixel(rec)
                if pix:
                    pts.append((int(pix[0]), int(pix[1])))
                    continue
            fx, fy = self._lot_to_frame(p['x'], p['y'], w, h)
            pts.append((fx, fy))

        # Glow shadow
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (100, 80, 0), 8, cv2.LINE_AA)

        # Animated dashed line
        for i in range(1, len(pts)):
            self._draw_dashed_line(frame, pts[i - 1], pts[i],
                                   COL_PATH, thickness=3,
                                   dash_len=18, gap_len=10,
                                   offset=self._anim_offset)

        # Waypoint dots
        for pt in pts[1:-1]:
            cv2.circle(frame, pt, 6, COL_PATH, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 8, (0, 0, 0), 1, cv2.LINE_AA)

        # Arrowhead at destination
        if len(pts) >= 2:
            self._draw_arrowhead(frame, pts[-2], pts[-1], COL_PATH, size=18)

        # Pulsing target ring
        if pts:
            dest = pts[-1]
            pulse_r = int(22 + 5 * math.sin(time.time() * 4))
            cv2.circle(frame, dest, pulse_r + 4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(frame, dest, pulse_r, COL_FREE, 2, cv2.LINE_AA)

            # "PARK HERE" label
            if rec:
                label = f"PARK: {rec}"
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
                lx = dest[0] - tw // 2
                ly = dest[1] - pulse_r - 14
                cv2.rectangle(frame, (lx - 6, ly - th - 4), (lx + tw + 6, ly + 4),
                              (0, 0, 0), -1)
                cv2.putText(frame, label, (lx, ly),
                            FONT, 0.55, COL_FREE, 1, cv2.LINE_AA)

        return frame

    @staticmethod
    def _draw_dashed_line(img, p1, p2, color, thickness=2,
                          dash_len=15, gap_len=10, offset=0):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        step = dash_len + gap_len
        pos = offset % step - step
        while pos < length:
            s = max(0, pos)
            e = min(length, pos + dash_len)
            if e > s:
                sx = int(p1[0] + ux * s)
                sy = int(p1[1] + uy * s)
                ex = int(p1[0] + ux * e)
                ey = int(p1[1] + uy * e)
                cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
            pos += step

    @staticmethod
    def _draw_arrowhead(img, p_from, p_to, color, size=16):
        angle = math.atan2(p_to[1] - p_from[1], p_to[0] - p_from[0])
        p1 = (int(p_to[0] - size * math.cos(angle - 0.45)),
              int(p_to[1] - size * math.sin(angle - 0.45)))
        p2 = (int(p_to[0] - size * math.cos(angle + 0.45)),
              int(p_to[1] - size * math.sin(angle + 0.45)))
        pts = np.array([p_to, p1, p2], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


# ── MAIN SYSTEM ───────────────────────────────────────────────────────────────
class ParkingSystem:
    """
    Complete pipeline:
      video → best.pt → 4-class detection
      → space mapping → space number overlay
      → A* path drawn on frame → HUD with VACANT/FULL counts
    """

    def __init__(self, weights="best.pt", api_url=None):
        print("[PARKING SYSTEM] Initializing...")
        self.model    = load_model(weights)
        self._weights = weights
        self.api_url  = api_url   # e.g. "http://localhost:5001/api"
        if api_url:
            print(f"[OK] API posting enabled → {api_url}")
        self._reset_state()
        

    def _reset_state(self):
        """Reset state between videos."""
        self.manager = ParkingManager()
        if hasattr(self, 'api_url'):
            self.manager.api_url = self.api_url
        self.mapper  = SpaceMapper(self.manager)
        self.path_ov = PathOverlay(self.manager, self.mapper)
        self.running = False
        self.frame_count = 0
        self._lock       = threading.Lock()
        self.latest_stats = {}

        self.det_colors = {
            0: (50,  100, 255),
            1: (0,   230, 80),
            2: (30,  30,  220),
            3: (0,   230, 80),
        }

    # ── PROCESS ONE FRAME ──────────────────────────────────────────────────────
    def process_frame(self, frame):
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

        # Auto-calibrate
        if not self.mapper.calibrated and self.frame_count <= 5:
            self.mapper.calibrate_from_detections(detections, w, h)

        # DEBUG: Print class distribution every 60 frames
        if self.frame_count % 60 == 1:
            cls_counts = {}
            for d in detections:
                c = d['cls']
                cls_counts[c] = cls_counts.get(c, 0) + 1
            free_count = sum(1 for d in detections if d['is_free'])
            busy_count = sum(1 for d in detections if not d['is_free'])
            print(f"  [DEBUG] Frame {self.frame_count}: Classes={cls_counts}  "
                  f"Free={free_count}  Busy={busy_count}  Total={len(detections)}")

        # Update manager
        yolo_updates = []
        for d in detections:
            if d['cls'] in {0, 1, 2, 3}:
                space_id = self.mapper.find_space_id(d['cx'], d['cy'], w, h)
                if space_id:
                    yolo_updates.append({
                        "space_id": space_id,
                        "is_free":  d['is_free'],
                        "confidence": d['conf'],
                    })
        self.manager.update_from_yolo(yolo_updates)

        # Send detections to Flask API (non-blocking)
        if self.api_url and yolo_updates and self.frame_count % 3 == 0:
            self._post_to_api(yolo_updates)

        # Navigation info
        rec_space = self.manager.find_nearest_free_space()
        nav_info  = self.manager.get_navigation_info(rec_space) if rec_space else {}

        # Build annotated frame
        annotated = frame.copy()
        annotated = self._draw_detections(annotated, detections)
        annotated = self.path_ov.draw(annotated, nav_info)
        annotated = self._draw_space_numbers(annotated, detections, w, h)
        annotated = self._draw_hud(annotated, nav_info)

        stats = self.manager.get_stats()
        with self._lock:
            self.latest_stats = stats
        return annotated, stats

    # ── POST TO FLASK API (background thread) ─────────────────────────────────
    def _post_to_api(self, yolo_updates):
        """Send YOLO detections to Flask API in a background thread."""
        if not _requests:
            return
        def _do_post():
            try:
                _requests.post(
                    f"{self.api_url}/batch_update",
                    json=yolo_updates,
                    timeout=1,
                )
            except Exception:
                pass  # Don't crash video processing if API is down
        threading.Thread(target=_do_post, daemon=True).start()

    def _post_video_info(self, video_name, video_idx, video_total):
        """Tell the Flask API which video is currently being processed."""
        if not self.api_url or not _requests:
            return
        def _do_post():
            try:
                _requests.post(
                    f"{self.api_url}/video_info",
                    json={"name": video_name, "index": video_idx, "total": video_total},
                    timeout=1,
                )
            except Exception:
                pass
        threading.Thread(target=_do_post, daemon=True).start()

    # ── DETECTION BOXES ───────────────────────────────────────────────────────
    def _draw_detections(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            color = self.det_colors.get(d['cls'], (200, 200, 200))
            label = f"{d['label']} {d['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, FONTM, 0.48, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        FONTM, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    # ── SPACE NUMBERS ─────────────────────────────────────────────────────────
    def _draw_space_numbers(self, frame, detections, fw, fh):
        """
        Draw space IDs (e.g. 'A03') boldly inside every detected space region.
        Outlined text ensures legibility on any background.
        Also draws the full space-map legend panel.
        """
        h, w = frame.shape[:2]

        for d in detections:
            if d['cls'] not in (1, 2, 3):
                continue
            x1, y1, x2, y2 = d['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            space_id = self.mapper.find_space_id(d['cx'], d['cy'], fw, fh)
            if not space_id:
                continue

            space   = self.manager.layout.spaces.get(space_id)
            is_free = space.is_free if space else d['is_free']
            color   = COL_FREE if is_free else COL_BUSY

            box_w = max(x2 - x1, 1)
            scale = max(0.5, min(1.1, box_w / 85))

            # Space ID — outlined
            tid_x = cx - int(len(space_id) * 9 * scale / 2)
            tid_y = cy + int(8 * scale)
            cv2.putText(frame, space_id, (tid_x, tid_y),
                        FONT, scale, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, space_id, (tid_x, tid_y),
                        FONT, scale, color, 1, cv2.LINE_AA)

            # Status chip
            chip = "FREE" if is_free else "BUSY"
            (cw, ch), _ = cv2.getTextSize(chip, FONTM, 0.34, 1)
            cv2.rectangle(frame,
                          (cx - cw // 2 - 3, y2 - ch - 6),
                          (cx + cw // 2 + 3, y2), color, -1)
            cv2.putText(frame, chip, (cx - cw // 2, y2 - 3),
                        FONTM, 0.34, (0, 0, 0), 1, cv2.LINE_AA)

        self._draw_space_legend(frame)
        return frame

    def _draw_space_legend(self, frame):
        """Compact top-right panel showing all space IDs colour-coded. Enlarged for readability."""
        h, w = frame.shape[:2]
        spaces = self.manager.layout.spaces
        if not spaces:
            return

        # Dynamically adjust columns so it doesn't get too tall if there are many spaces
        cols   = max(4, math.ceil(len(spaces) / 10)) 
        if cols > 8: cols = 8

        cell_w = 85
        cell_h = 32
        pad    = 15
        rows   = math.ceil(len(spaces) / cols)
        p_w    = cols * cell_w + pad * 2
        p_h    = rows * cell_h + pad * 2 + 35
        px     = w - p_w - 15
        py     = 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + p_w, py + p_h), COL_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (px, py), (px + p_w, py + p_h), (80, 120, 180), 2)
        cv2.putText(frame, "SPACE LOG", (px + pad, py + 26),
                    FONTM, 0.65, (180, 220, 255), 2, cv2.LINE_AA)

        for i, (sid, space) in enumerate(sorted(spaces.items())):
            col = i % cols
            row = i // cols
            cx2 = px + pad + col * cell_w
            cy2 = py + 40 + pad + row * cell_h
            color = COL_FREE if space.is_free else COL_BUSY
            
            # Draw larger space chip
            cv2.rectangle(frame, (cx2, cy2 - 20), (cx2 + cell_w - 6, cy2 + 10), color, -1)
            cv2.rectangle(frame, (cx2, cy2 - 20), (cx2 + cell_w - 6, cy2 + 10), (0, 0, 0), 2)
            
            # Draw larger bold text
            cv2.putText(frame, sid, (cx2 + 12, cy2 + 3),
                        FONTM, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────────────────────
    def _draw_hud(self, frame, nav_info):
        """Main left-side HUD with large VACANT / OCCUPIED counters."""
        stats = self.manager.get_stats()
        rec   = nav_info.get("recommended_space", "FULL")
        h, w  = frame.shape[:2]

        # Background panel
        p_w, p_h = 320, 215
        overlay = frame.copy()
        cv2.rectangle(overlay, (6, 6), (6 + p_w, 6 + p_h), COL_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (6, 6), (6 + p_w, 6 + p_h), (60, 120, 200), 2)

        y = 30
        cv2.putText(frame, "PARKVISION", (16, y), FONT, 0.65, (0, 210, 255), 1, cv2.LINE_AA)
        y += 8
        cv2.line(frame, (16, y), (6 + p_w - 10, y), (60, 100, 160), 1)
        y += 18

        # VACANT label + big number
        cv2.putText(frame, "VACANT", (16, y), FONTM, 0.50, (150, 200, 150), 1, cv2.LINE_AA)
        vac_txt = str(stats['free'])
        cv2.putText(frame, vac_txt, (16, y + 46), FONT, 2.0, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(frame, vac_txt, (16, y + 46), FONT, 2.0, COL_FREE,    2, cv2.LINE_AA)

        # OCCUPIED label + big number (right side)
        cv2.putText(frame, "OCCUPIED", (175, y), FONTM, 0.50, (200, 150, 150), 1, cv2.LINE_AA)
        occ_txt = str(stats['busy'])
        cv2.putText(frame, occ_txt, (175, y + 46), FONT, 2.0, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(frame, occ_txt, (175, y + 46), FONT, 2.0, COL_BUSY,   2, cv2.LINE_AA)
        y += 58

        # Total + occupancy %
        cv2.putText(frame, f"TOTAL: {stats['total']}   OCC: {stats['occupancy_pct']}%",
                    (16, y + 8), FONTM, 0.50, (200, 220, 255), 1, cv2.LINE_AA)
        y += 22

        # Occupancy bar
        bx, bw2, bh2 = 16, p_w - 26, 12
        filled = int(bw2 * stats['busy'] / max(stats['total'], 1))
        cv2.rectangle(frame, (bx, y), (bx + bw2, y + bh2), (40, 40, 60), -1)
        bar_c = COL_FREE if stats['free'] > 5 else (0, 165, 255) if stats['free'] > 2 else COL_BUSY
        cv2.rectangle(frame, (bx, y), (bx + filled, y + bh2), bar_c, -1)
        cv2.rectangle(frame, (bx, y), (bx + bw2, y + bh2), (80, 120, 180), 1)

        # Driver guide panel (below main HUD)
        if rec and rec != "FULL" and stats['free'] > 0:
            gy = 6 + p_h + 10
            g_h = 105
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (6, gy), (6 + p_w, gy + g_h), (20, 35, 55), -1)
            cv2.addWeighted(overlay2, 0.80, frame, 0.20, 0, frame)
            cv2.rectangle(frame, (6, gy), (6 + p_w, gy + g_h), (0, 180, 80), 2)

            ny = gy + 20
            cv2.putText(frame, "A* DRIVER GUIDE", (16, ny),
                        FONTM, 0.52, (0, 220, 160), 1, cv2.LINE_AA)
            ny += 26
            cv2.putText(frame, "GO TO SPACE:", (16, ny),
                        FONTM, 0.48, (180, 220, 255), 1, cv2.LINE_AA)
            # Target space ID — large + prominent
            cv2.putText(frame, rec, (145, ny + 4), FONT, 0.90, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(frame, rec, (145, ny + 4), FONT, 0.90, COL_FREE,   2, cv2.LINE_AA)
            ny += 28
            walk_t  = nav_info.get('estimated_walk_seconds', 0)
            sp_info = self.manager.layout.spaces.get(rec)
            section = sp_info.section if sp_info else "?"
            cv2.putText(frame, f"Section {section}   Walk: ~{walk_t}s",
                        (16, ny), FONTM, 0.46, (160, 200, 160), 1, cv2.LINE_AA)

        # Frame counter (bottom-right)
        cv2.putText(frame, f"#{self.frame_count}", (w - 75, h - 12),
                    FONTM, 0.38, (80, 120, 160), 1, cv2.LINE_AA)

        return frame

    # ── GATE TRIGGER ─────────────────────────────────────────────────────────
    def on_vehicle_at_gate(self, prefer_section=None):
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

    # ── SINGLE WINDOW NAME (shared across all videos) ─────────────────────────
    WINDOW_NAME = "ParkVision"

    # ── RUN ONE VIDEO ─────────────────────────────────────────────────────────
    def run_video(self, video_path, display=True, gate_sim_every=90,
                  video_idx=1, video_total=1):
        """
        Process a single video from start to finish.
        Uses a SINGLE shared window (created externally) for display.
        Returns final stats dict.
        Press Q to quit, N to skip to next video.
        """
        self._reset_state()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {video_path}")
            return None

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n{'='*70}")
        print(f"  VIDEO : {Path(video_path).name}")
        print(f"  Size  : {vw}x{vh}   FPS: {fps:.0f}   Frames: {total_frames}")
        print(f"{'='*70}")

        if display:
            print("  [Keys] Q = Quit all   N = Next video")

        # Video banner text (shown on each frame)
        banner_text = f"[{video_idx}/{video_total}] {Path(video_path).name}"

        # Notify API about current video
        self._post_video_info(Path(video_path).name, video_idx, video_total)

        self.running = True
        skip_to_next = False

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"  [DONE] Reached end of video.")
                break

            annotated, stats = self.process_frame(frame)

            # Draw video-name banner at the bottom so user knows which video
            annotated = self._draw_video_banner(annotated, banner_text)

            if gate_sim_every and self.frame_count % gate_sim_every == 0:
                self.on_vehicle_at_gate()

            if self.frame_count % 100 == 0:
                pct = self.frame_count / max(total_frames, 1) * 100
                print(f"  [{pct:5.1f}%] Frame {self.frame_count}/{total_frames} | "
                      f"Vacant: {stats['free']}  Occupied: {stats['busy']}")

            if display:
                cv2.imshow(self.WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                
                # Check if user closed the window via the X button
                try:
                    win_visible = cv2.getWindowProperty(
                        self.WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                    if win_visible < 1:
                        print("  [SKIP] Window closed — moving to next video.")
                        skip_to_next = True
                        break
                except cv2.error:
                    print("  [SKIP] Window closed — moving to next video.")
                    skip_to_next = True
                    break

                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('n'):
                    skip_to_next = True
                    break

        cap.release()

        final_stats = self.manager.get_stats()
        self._generate_summary_report(Path(video_path).name)

        if skip_to_next:
            return final_stats, "skip"
        return final_stats, "done" if self.running else "quit"

    # ── VIDEO BANNER ──────────────────────────────────────────────────────────
    def _draw_video_banner(self, frame, text):
        """Draw a small banner at the bottom-left showing the current video name."""
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, FONTM, 0.50, 1)
        bx, by = 6, h - 8
        cv2.rectangle(frame, (bx - 4, by - th - 8), (bx + tw + 8, by + 4),
                      COL_HUD_BG, -1)
        cv2.rectangle(frame, (bx - 4, by - th - 8), (bx + tw + 8, by + 4),
                      (60, 120, 200), 1)
        cv2.putText(frame, text, (bx, by - 3),
                    FONTM, 0.50, (180, 220, 255), 1, cv2.LINE_AA)
        return frame

    # ── SUMMARY REPORT ────────────────────────────────────────────────────────
    def _generate_summary_report(self, video_name=""):
        stats  = self.manager.get_stats()
        spaces = self.manager.layout.spaces
        vacant   = [s for s in spaces.values() if s.is_free]
        occupied = [s for s in spaces.values() if not s.is_free]

        print(f"\n{'='*70}")
        print(f"  ANALYSIS REPORT — {video_name}")
        print(f"{'='*70}")
        print(f"  Total Spaces   : {stats['total']}")
        print(f"  VACANT         : {stats['free']}  ({stats['free']/max(stats['total'],1)*100:.1f}%)")
        print(f"  OCCUPIED       : {stats['busy']}  ({stats['busy']/max(stats['total'],1)*100:.1f}%)")
        print(f"  Occupancy Rate : {stats['occupancy_pct']}%")

        if vacant:
            by_sec = {}
            for s in vacant:
                by_sec.setdefault(s.section, []).append(s.id)
            print(f"\n  VACANT SPACES:")
            for sec in sorted(by_sec):
                print(f"    Section {sec}: {', '.join(sorted(by_sec[sec]))}")
        else:
            print("\n  WARNING — LOT IS FULL (no vacant spaces)")

        if stats['free'] > 0:
            nearest  = self.manager.find_nearest_free_space()
            nav_info = self.manager.get_navigation_info(nearest)
            print(f"\n  A* NAVIGATION:")
            print(f"    Recommended Space : {nearest}")
            print(f"    Section           : {nav_info.get('section')}")
            print(f"    Walk time         : ~{nav_info.get('estimated_walk_seconds')}s")
        print(f"{'='*70}\n")


# ── SEQUENTIAL MULTI-VIDEO RUNNER ─────────────────────────────────────────────
def run_videos_sequentially(video_paths, weights="best.pt", display=True,
                            api_url=None):
    """
    Process a list of video files ONE BY ONE in a SINGLE OpenCV window.
    The window is created once and reused for every video.
    If api_url is set, detections are streamed to the Flask API.
    """
    if not video_paths:
        print("[ERROR] No video files provided.")
        return

    print(f"\n{'#'*70}")
    print(f"  PARKVISION — Processing {len(video_paths)} video(s) sequentially")
    for i, v in enumerate(video_paths, 1):
        print(f"    {i:2d}. {Path(v).name}")
    print(f"{'#'*70}\n")

    system      = ParkingSystem(weights=weights, api_url=api_url)
    all_results = []

    for idx, vpath in enumerate(video_paths, 1):
        # ── Ensure the window exists (create/re-create if user closed it) ─────
        if display:
            try:
                vis = cv2.getWindowProperty(
                    ParkingSystem.WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                if vis < 1:
                    raise cv2.error
            except cv2.error:
                cv2.namedWindow(ParkingSystem.WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(ParkingSystem.WINDOW_NAME, 1280, 720)

        print(f"\n>>> VIDEO {idx}/{len(video_paths)}: {Path(vpath).name}")
        result = system.run_video(str(vpath), display=display,
                                  video_idx=idx, video_total=len(video_paths))

        if isinstance(result, tuple):
            stats, status = result
        else:
            stats, status = result, "done"

        if stats:
            all_results.append({
                "video": Path(vpath).name,
                "stats": stats,
                "status": status,
            })

        if status == "quit":
            print("\n[QUIT] User pressed Q — stopping all videos.")
            break

    # ── Destroy the single window after ALL videos ────────────────────────────
    if display:
        cv2.destroyAllWindows()

    # Final combined summary
    print(f"\n{'#'*70}")
    print(f"  ALL DONE — {len(all_results)}/{len(video_paths)} video(s) processed")
    print(f"{'#'*70}")
    print(f"  {'Video':<42} {'Vacant':>7} {'Occupied':>9} {'Occ%':>6}")
    print(f"  {'-'*65}")
    for r in all_results:
        s = r['stats']
        flag = " [skipped]" if r['status'] == "skip" else ""
        print(f"  {r['video']:<42} {s['free']:>7} {s['busy']:>9} {s['occupancy_pct']:>5}%{flag}")
    print()


# ── FIND VIDEO FILES ──────────────────────────────────────────────────────────
def collect_videos(folder=None, sources=None):
    """Return ordered list of video file paths."""
    EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    results = []

    if sources:
        for s in sources:
            p = Path(s)
            if p.is_file() and p.suffix.lower() in EXTS:
                results.append(p)
            elif p.is_dir():
                for ext in EXTS:
                    results.extend(sorted(p.glob(f"*{ext}")))
            else:
                print(f"[WARN] Not found or unsupported: {s}")

    if folder:
        fp = Path(folder)
        if fp.is_dir():
            for ext in EXTS:
                results.extend(sorted(fp.glob(f"*{ext}")))
        else:
            print(f"[WARN] Folder not found: {folder}")

    # Auto-search fallback
    if not results:
        for d in ["Car-Videos", "car-videos", "videos", "."]:
            for ext in EXTS:
                results.extend(sorted(Path(d).glob(f"*{ext}")))
            if results:
                break

    # Deduplicate, preserve order
    seen, unique = set(), []
    for p in results:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ParkVision — Sequential parking analysis with A* driver guidance",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--source", nargs="+", default=None,
        help="One or more video files (processed one by one)",
    )
    ap.add_argument(
        "--folder", default=None,
        help="Folder of video files (all processed one by one)",
    )
    ap.add_argument(
        "--weights", default="best.pt",
        help="YOLO weights file (default: best.pt)",
    )
    ap.add_argument(
        "--headless", action="store_true",
        help="No display window (faster, good for servers)",
    )
    ap.add_argument("--api", action="store_true", default=True,
                        help="Send detections to local Flask API (http://localhost:5001/api)")
    ap.add_argument(
        "--api-url", default="http://localhost:5001/api",
        help="Flask API base URL (default: http://localhost:5001/api)",
    )
    args = ap.parse_args()

    videos = collect_videos(folder=args.folder, sources=args.source)

    if not videos:
        print("\n[ERROR] No video files found.")
        print("  Options:")
        print("  1. Put .mp4 files in a 'Car-Videos/' folder")
        print("  2. python parking_integration.py --source video.mp4")
        print("  3. python parking_integration.py --folder Car-Videos/")
        sys.exit(1)

    api_url = args.api_url if args.api else None
    if api_url:
        print(f"[API] Will stream detections to {api_url}")
        print(f"[API] Open dashboard: http://localhost:5001/dashboard")

    run_videos_sequentially(
        video_paths=videos,
        weights=args.weights,
        display=not args.headless,
        api_url=api_url,
    )
