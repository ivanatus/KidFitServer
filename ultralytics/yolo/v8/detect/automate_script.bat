@echo off
setlocal enabledelayedexpansion

:loop
REM Set the URL to your Firebase Realtime Database and the path to the service account key JSON file
set "firebase_url=https://exercise-project-bc0dc-default-rtdb.europe-west1.firebasedatabase.app/Unanalyzed.json"
set "service_account_key=C:/Users/Ivana/Documents/GitHub/Exercise-Project/ARCHIVE/2023-10-27/ExerciseProject - Python/ultralytics/yolo/v8/detect/serviceAccountKey.json"

REM Use Python to fetch the data from Firebase
python -c "import requests; import json; response = requests.get('!firebase_url!', headers={'Content-Type': 'application/json'}, data=open('!service_account_key!', 'rb').read()); data = response.text.strip()" > response.txt

REM Check if the "Unanalyzed" flag is set to "true" (assuming 'true' means unanalyzed)
set /p unanalyzed_flag=<response.txt

if "!unanalyzed_flag!"=="true" (
  REM If the flag is true, trigger the Python script
  python "C:/Users/Ivana/Documents/GitHub/Exercise-Project/ARCHIVE/2023-10-27/ExerciseProject - Python/ultralytics/yolo/v8/detect/predict.py" model=yolov8l.pt show=True
)

REM Delay for 30 minutes (1800 seconds) before checking again
timeout /t 1800 /nobreak
goto loop

