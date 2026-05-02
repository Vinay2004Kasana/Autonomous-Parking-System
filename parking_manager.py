"""
parking_manager.py - Core parking space manager with real-time tracking and path planning
Integrates with your existing YOLO-based detection from model.ipynb
"""

import json
import time
import math
import threading
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from collections import deque


@dataclass
class ParkingSpace:
    id: str
    row: int
    col: int
    x: float          # pixel x center
    y: float          # pixel y center
    width: float
    height: float
    is_free: bool
    section: str      # e.g., "A", "B", "C"
    last_updated: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class PathNode:
    x: float
    y: float
    node_type: str    # "entry", "lane", "space"
    space_id: Optional[str] = None


class ParkingLayout:
    """
    Represents the physical parking lot layout.
    Adapt the grid dimensions to match your actual parking lot.
    """
    def __init__(self, rows=4, cols=6, lot_width=800, lot_height=600):
        self.rows = rows
        self.cols = cols
        self.lot_width = lot_width
        self.lot_height = lot_height
        self.entry_point = (50, lot_height // 2)    # Gate/entry position
        self.spaces: Dict[str, ParkingSpace] = {}
        self.init_dynamic_layout(rows * cols)

    def init_dynamic_layout(self, total_spaces: int):
        """Dynamically generate a grid for the given number of spaces."""
        self.spaces.clear()
        if total_spaces <= 0:
            return
            
        # Calculate optimal grid dimensions to fit on screen
        self.cols = math.ceil(math.sqrt(total_spaces * 1.5))
        self.rows = math.ceil(total_spaces / self.cols)
        
        margin_x = 120
        margin_y = 80
        spacing_x = (self.lot_width - 2 * margin_x) / max(self.cols - 1, 1)
        spacing_y = (self.lot_height - 2 * margin_y) / max(self.rows - 1, 1)

        section_labels = ["A", "B", "C", "D", "E", "F"]
        space_w, space_h = 55, 35

        for i in range(total_spaces):
            r = i // self.cols
            c = i % self.cols
            section = section_labels[r % len(section_labels)]
            sid = f"{section}{i + 1:02d}"
            cx = margin_x + c * spacing_x
            cy = margin_y + r * spacing_y
            self.spaces[sid] = ParkingSpace(
                id=sid,
                row=r,
                col=c,
                x=cx,
                y=cy,
                width=space_w,
                height=space_h,
                is_free=True,
                section=section,
                last_updated=time.time()
            )


class PathPlanner:
    """
    A* path planner for navigating from the parking gate to a free space.
    Works on a grid representation of the parking lot.
    """

    def __init__(self, layout: ParkingLayout, grid_res=20):
        self.layout = layout
        self.grid_res = grid_res    # pixels per grid cell
        self.grid_w = math.ceil(layout.lot_width / grid_res)
        self.grid_h = math.ceil(layout.lot_height / grid_res)

    def _pixel_to_grid(self, px, py) -> Tuple[int, int]:
        return int(px / self.grid_res), int(py / self.grid_res)

    def _grid_to_pixel(self, gx, gy) -> Tuple[float, float]:
        return gx * self.grid_res + self.grid_res / 2, gy * self.grid_res + self.grid_res / 2

    def _build_obstacle_grid(self, target_space_id: str):
        """Mark occupied spaces as obstacles (except the target space)"""
        occupied = set()
        for sid, space in self.layout.spaces.items():
            if sid != target_space_id and not space.is_free:
                gx, gy = self._pixel_to_grid(space.x, space.y)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        occupied.add((gx + dx, gy + dy))
        return occupied

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def find_path(self, target_space_id: str) -> List[Tuple[float, float]]:
        """Return list of (x, y) pixel waypoints from entry to target space."""
        target = self.layout.spaces.get(target_space_id)
        if not target:
            return []

        obstacles = self._build_obstacle_grid(target_space_id)
        start = self._pixel_to_grid(*self.layout.entry_point)
        goal = self._pixel_to_grid(target.x, target.y)

        # A* search
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(self._grid_to_pixel(*current))
                    current = came_from[current]
                path.append(self._grid_to_pixel(*start))
                path.reverse()
                return self._smooth_path(path)

            open_set.remove(current)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (neighbor[0] < 0 or neighbor[0] >= self.grid_w or
                        neighbor[1] < 0 or neighbor[1] >= self.grid_h):
                    continue
                if neighbor in obstacles:
                    continue
                tentative_g = g_score.get(current, float('inf')) + self._heuristic(current, neighbor)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    open_set.add(neighbor)

        # Fallback: direct path with waypoints
        return self._direct_path(target)

    def _direct_path(self, space: ParkingSpace) -> List[Tuple[float, float]]:
        """Fallback direct path with intermediate lane waypoints."""
        ex, ey = self.layout.entry_point
        lane_x = space.x - 80
        return [
            (ex, ey),
            (lane_x, ey),
            (lane_x, space.y),
            (space.x, space.y)
        ]

    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Reduce redundant waypoints."""
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            px, py = path[i - 1]
            cx, cy = path[i]
            nx, ny = path[i + 1]
            if not (abs(cx - px) < 5 and abs(cx - nx) < 5) and \
               not (abs(cy - py) < 5 and abs(cy - ny) < 5):
                smoothed.append(path[i])
        smoothed.append(path[-1])
        return smoothed

    def calculate_distance(self, path: List[Tuple[float, float]]) -> float:
        total = 0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total


class ParkingManager:
    """
    Main parking manager. Connect your YOLO detection output here
    by calling update_from_yolo() with detection results.
    """

    def __init__(self):
        self.layout = ParkingLayout(rows=4, cols=6)
        self.path_planner = PathPlanner(self.layout)
        self._lock = threading.Lock()
        self.event_log = deque(maxlen=50)
        self._simulate_initial_state()

    def reset(self):
        """Reset all spaces to free (call between videos for a clean slate)."""
        with self._lock:
            for space in self.layout.spaces.values():
                space.is_free = True
                space.last_updated = time.time()
            self.event_log.clear()

    def init_dynamic_layout(self, total_spaces: int):
        """Rebuild the parking layout dynamically."""
        with self._lock:
            self.layout.init_dynamic_layout(total_spaces)
            self.event_log.clear()
            self._log_event(f"System auto-calibrated with {total_spaces} spaces.")

    def _simulate_initial_state(self):
        """Pre-fill some spaces as occupied for demo purposes."""
        occupied_ids = ["A01", "A02", "A04", "A07", "B13", "B14", "B16", "B19", "B20", "B22"]
        for sid in occupied_ids:
            if sid in self.layout.spaces:
                self.layout.spaces[sid].is_free = False

    # ─────────────────────────────────────────────
    # YOLO INTEGRATION POINT
    # ─────────────────────────────────────────────
    def update_from_yolo(self, detections: List[dict]):
        """
        Call this with your YOLO model output.

        Each detection dict should contain:
        {
            "space_id": "A01",          # or map bbox to nearest space
            "is_free": True/False,
            "confidence": 0.95
        }

        In your model.ipynb / app.py, after running inference:
            detections = []
            for result in yolo_results:
                cls = int(result.boxes.cls[0])
                # cls 0 = free, cls 1 = occupied (adjust to your model)
                detections.append({
                    "space_id": map_bbox_to_space(result.boxes.xyxy[0]),
                    "is_free": cls == 0,
                    "confidence": float(result.boxes.conf[0])
                })
            manager.update_from_yolo(detections)
        """
        with self._lock:
            for det in detections:
                sid = det.get("space_id")
                if sid and sid in self.layout.spaces:
                    space = self.layout.spaces[sid]
                    old_status = space.is_free
                    space.is_free = det.get("is_free", True)
                    space.last_updated = time.time()
                    if old_status != space.is_free:
                        event = "freed" if space.is_free else "occupied"
                        self._log_event(f"Space {sid} {event}")

    def update_space(self, space_id: str, is_free: bool):
        """Manually update a single space (useful for gate sensors)."""
        with self._lock:
            if space_id in self.layout.spaces:
                self.layout.spaces[space_id].is_free = is_free
                self.layout.spaces[space_id].last_updated = time.time()
                self._log_event(f"Space {space_id} {'freed' if is_free else 'occupied'}")

    def get_stats(self) -> dict:
        with self._lock:
            spaces = list(self.layout.spaces.values())
            free = [s for s in spaces if s.is_free]
            busy = [s for s in spaces if not s.is_free]
            sections = {}
            for s in spaces:
                sec = s.section
                if sec not in sections:
                    sections[sec] = {"free": 0, "busy": 0}
                if s.is_free:
                    sections[sec]["free"] += 1
                else:
                    sections[sec]["busy"] += 1
            return {
                "total": len(spaces),
                "free": len(free),
                "busy": len(busy),
                "occupancy_pct": round(len(busy) / max(len(spaces), 1) * 100, 1),
                "sections": sections,
            }

    def find_nearest_free_space(self, prefer_section: Optional[str] = None) -> Optional[str]:
        """Find the closest free space to the entry gate."""
        with self._lock:
            free_spaces = [s for s in self.layout.spaces.values() if s.is_free]
            if not free_spaces:
                return None
            if prefer_section:
                preferred = [s for s in free_spaces if s.section == prefer_section]
                if preferred:
                    free_spaces = preferred
            ex, ey = self.layout.entry_point
            return min(free_spaces,
                       key=lambda s: math.sqrt((s.x - ex) ** 2 + (s.y - ey) ** 2)).id

    def get_navigation_info(self, space_id: Optional[str] = None) -> dict:
        """Get full navigation package for the driver display."""
        if not space_id:
            space_id = self.find_nearest_free_space()
        if not space_id:
            return {"error": "No free spaces available"}

        path = self.path_planner.find_path(space_id)
        distance = self.path_planner.calculate_distance(path)
        space = self.layout.spaces[space_id]
        stats = self.get_stats()

        return {
            "recommended_space": space_id,
            "section": space.section,
            "path": [{"x": p[0], "y": p[1]} for p in path],
            "distance_px": round(distance, 1),
            "estimated_walk_seconds": round(distance / 50),  # ~50px per second walking speed
            "stats": stats,
            "all_spaces": [s.to_dict() for s in self.layout.spaces.values()],
            "entry_point": {"x": self.layout.entry_point[0], "y": self.layout.entry_point[1]},
            "lot_dimensions": {"width": self.layout.lot_width, "height": self.layout.lot_height},
        }

    def _log_event(self, msg: str):
        self.event_log.appendleft({"time": time.strftime("%H:%M:%S"), "msg": msg})

    def get_event_log(self) -> list:
        return list(self.event_log)


# ─────────────────────────────────────────────────────────────────
# FLASK API SERVER — serves real-time data to parking_ui.html
# Receives YOLO detections from parking_integration.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from flask import Flask, jsonify, request, send_file
        from flask_cors import CORS
    except ImportError:
        print("Install: pip install flask flask-cors")
        exit(1)

    import os

    app = Flask(__name__)
    CORS(app)
    manager = ParkingManager()

    # Track which video is currently being processed
    _current_video = {"name": "Waiting for video feed...", "index": 0, "total": 0}

    @app.route("/")
    def root():
        """Root endpoint with API documentation"""
        return jsonify({
            "service": "Parking Management API",
            "version": "2.0",
            "status": "running",
            "mode": "LIVE — waiting for YOLO detections from parking_integration.py",
            "dashboard": "http://localhost:5001/dashboard",
            "endpoints": {
                "GET /dashboard": "Open the live parking dashboard in your browser",
                "GET /api/status": "Get current parking lot statistics",
                "GET /api/navigate": "Get navigation info to nearest free space",
                "GET /api/spaces": "Get all parking spaces with their current status",
                "GET /api/events": "Get recent parking event log",
                "GET /api/video_info": "Get current video being processed",
                "POST /api/update": "Update a space status (JSON: {space_id, is_free})",
                "POST /api/batch_update": "Batch update from YOLO (JSON: [{space_id, is_free, confidence}, ...])",
                "POST /api/video_info": "Set current video info (JSON: {name, index, total})",
            },
        })

    @app.route("/dashboard")
    def dashboard():
        """Serve the parking_ui.html dashboard directly."""
        html_path = os.path.join(os.path.dirname(__file__), "parking_ui.html")
        if os.path.exists(html_path):
            return send_file(html_path)
        return "parking_ui.html not found", 404

    @app.route("/api/status")
    def status():
        return jsonify(manager.get_stats())

    @app.route("/api/navigate")
    def navigate():
        space_id = request.args.get("space_id")
        section = request.args.get("section")
        if not space_id:
            space_id = manager.find_nearest_free_space(prefer_section=section)
        nav = manager.get_navigation_info(space_id)
        # Also include events so dashboard gets everything in one call
        nav["events"] = manager.get_event_log()
        return jsonify(nav)

    @app.route("/api/spaces")
    def spaces():
        with manager._lock:
            return jsonify([s.to_dict() for s in manager.layout.spaces.values()])

    @app.route("/api/update", methods=["POST"])
    def update():
        data = request.json
        manager.update_space(data["space_id"], data["is_free"])
        return jsonify({"ok": True})

    @app.route("/api/batch_update", methods=["POST"])
    def batch_update():
        """Receive YOLO detection batch from parking_integration.py"""
        data = request.json
        if isinstance(data, list):
            manager.update_from_yolo(data)
        return jsonify({"ok": True, "updated": len(data) if isinstance(data, list) else 0})

    @app.route("/api/init_spaces", methods=["POST"])
    def init_spaces():
        """Initialize the layout with a dynamic number of spaces."""
        data = request.json
        if "total" in data:
            manager.init_dynamic_layout(data["total"])
        return jsonify({"ok": True})

    @app.route("/api/video_info", methods=["GET", "POST"])
    def video_info():
        if request.method == "POST":
            info = request.json
            _current_video.update(info)
            return jsonify({"ok": True})
        return jsonify(_current_video)

    @app.route("/api/events")
    def events():
        return jsonify(manager.get_event_log())

    print("=" * 60)
    print("  PARKVISION API SERVER")
    print("  API     : http://localhost:5001")
    print("  Dashboard: http://localhost:5001/dashboard")
    print("=" * 60)
    print("  Waiting for YOLO detections from parking_integration.py...")
    print("  Run: python parking_integration.py --folder Car-Videos/ --api")
    print("=" * 60)
    app.run(port=5001, debug=False)
