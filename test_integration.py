"""Quick 5-frame test to check for runtime errors in parking_integration.py"""
import cv2
import sys
import traceback

from parking_integration import ParkingSystem

system = ParkingSystem()
cap = cv2.VideoCapture("Car-Videos/4208194-uhd_3840_2160_24fps (1).mp4")
if not cap.isOpened():
    print("ERROR: cannot open video")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {w}x{h} @ {fps}fps, {total} frames")

# Process only 5 frames to check for errors
for i in range(5):
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {i}")
        break
    try:
        annotated, stats = system.process_frame(frame)
        free = stats["free"]
        busy = stats["busy"]
        print(f"Frame {i+1}: OK | Free={free} Busy={busy}")
    except Exception as e:
        print(f"Frame {i+1}: ERROR -> {type(e).__name__}: {e}")
        traceback.print_exc()

cap.release()
print("DONE - 5-frame test complete")
