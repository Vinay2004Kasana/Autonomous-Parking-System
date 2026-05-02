@echo off
echo Starting Parking Manager (Backend API)...
start cmd /k ".venv\Scripts\activate && python parking_manager.py"

echo Starting Parking Integration (YOLO Vision Pipeline)...
start cmd /k ".venv\Scripts\activate && python parking_integration.py"

echo Opening Parking UI in your default browser...
start "" "parking_ui.html"

echo All components have been launched!
