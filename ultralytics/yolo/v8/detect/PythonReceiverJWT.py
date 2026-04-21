from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Header, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from datetime import datetime, timedelta, timezone
import uvicorn
import sys
import queue
import threading
import shutil
import time
import traceback
from uuid import uuid4
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "LEA_Python"))
import LEAdecryptCBC
import LEAdecryptCTR
import pandas as pd
import subprocess

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi import Request
from slowapi.errors import RateLimitExceeded

# App setup
app = FastAPI()
UPLOAD_FOLDER = "video"
RESULT_FOLDER = "results"
UPLOAD_DIR = os.path.join(BASE_DIR, UPLOAD_FOLDER)
RESULT_DIR = os.path.join(BASE_DIR, RESULT_FOLDER)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
revoked_tokens = set()
video_job_queue = queue.Queue()
video_jobs = {}
video_jobs_lock = threading.Lock()
worker_thread = None
STOP_SENTINEL = object()
# CV runtime knobs (change these to trade speed vs accuracy)
CV_MODEL = "yolov8s.pt"
CV_IMGSZ = 640

# Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
# Decrypt and get user database from csv
def get_user_database():
    LEAdecryptCBC.decrypt_file("DataBase.enc", "DataBase.csv")
    df = pd.read_csv("DataBase.csv", encoding="latin1")
    df.columns = [col.strip().lower() for col in df.columns]
    df["password"] = df["password"].astype(str)

    fake_users_db = {
        row["username"]: {
            "username": row["username"],
            "UID": str(row["uid"]),
            "hashed_password": pwd_context.hash(str(row["password"]))
        }
        for _, row in df.iterrows()
    }
    os.remove("DataBase.csv")
    return fake_users_db

# Secret key for JWT
SECRET_KEY = "my_jwt_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60*24*7 #acces token expires after a week

# Password hashing
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str):
    fake_users_db = get_user_database()
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta=None):
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    if token in revoked_tokens:
        raise HTTPException(status_code=401, detail="Token has been revoked")
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        fake_users_db = get_user_database()
        user = fake_users_db.get(username)
        if user is None:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def set_job_state(job_id, **updates):
    with video_jobs_lock:
        if job_id in video_jobs:
            video_jobs[job_id].update(updates)


def analyze_video():
    """Run predict.py on all of the videos from video folder"""
    predict_script_path = os.path.join(BASE_DIR, "predict.py")
    predict_command = [
        "python",
        predict_script_path,
        f"model={CV_MODEL}",
        f"imgsz={CV_IMGSZ}",
        "show=False",
        "source=video",
        "save=False",
        "save_txt=False"
    ]
    print("Start analysis")
    print(predict_command)
    return subprocess.run(predict_command, cwd=BASE_DIR, capture_output=True, text=True)


def video_worker():
    print("[video-worker] Worker thread is running and waiting for jobs")
    while True:
        job = video_job_queue.get()
        if job is STOP_SENTINEL:
            print("[video-worker] Stop signal received, worker exiting")
            video_job_queue.task_done()
            break

        job_id = job["job_id"]
        encrypted_file = job["encrypted_file"]
        decrypted_file = job["decrypted_file"]
        username = job["username"]
        user_uid = job.get("user_uid")
        set_job_state(job_id, status="running", started_at=now_iso())
        print(f"[video-worker] Job started: {job_id} ({os.path.basename(encrypted_file)})")

        try:
            LEAdecryptCTR.decrypt_video(encrypted_file, decrypted_file)
            if os.path.exists(encrypted_file):
                os.remove(encrypted_file)

            cv_start = time.perf_counter()
            result = analyze_video()
            cv_duration_sec = round(time.perf_counter() - cv_start, 3)
            print(f"[video-worker] CV processing time: {cv_duration_sec}s (job: {job_id})")
            # Print only the final CSV dump block from predict.py output.
            if result.stdout:
                marker_start = "=== UID_CSV_BEFORE_ENCRYPTION_START ==="
                marker_end = "=== UID_CSV_BEFORE_ENCRYPTION_END ==="
                start_idx = result.stdout.find(marker_start)
                end_idx = result.stdout.find(marker_end)
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    end_idx += len(marker_end)
                    print(result.stdout[start_idx:end_idx])
            if result.returncode != 0:
                if result.stderr:
                    print(result.stderr, end="")
                error_msg = (result.stderr or result.stdout or "predict.py failed").strip()
                raise RuntimeError(error_msg)

            # predict.py may produce <video_prefix>.enc (often UID-based) in results/ or BASE_DIR.
            # Normalize destination to results/<uid>.enc (fallback: username).
            video_prefix = os.path.splitext(os.path.basename(decrypted_file))[0].split("_")[0]
            result_stem = user_uid if user_uid else username
            src_enc_candidates = [
                os.path.join(RESULT_DIR, f"{video_prefix}.enc"),
                os.path.join(RESULT_DIR, f"{result_stem}.enc"),
                os.path.join(RESULT_DIR, f"{username}.enc"),
                os.path.join(BASE_DIR, f"{video_prefix}.enc"),
                os.path.join(BASE_DIR, f"{result_stem}.enc"),
                os.path.join(BASE_DIR, f"{username}.enc"),
            ]
            src_enc = next((p for p in src_enc_candidates if os.path.exists(p)), None)
            if src_enc is None:
                raise FileNotFoundError(f"Encrypted result not found for job {job_id}")

            dst_enc = os.path.join(RESULT_DIR, f"{result_stem}.enc")
            if os.path.abspath(src_enc) != os.path.abspath(dst_enc) and os.path.exists(dst_enc):
                os.remove(dst_enc)
            if os.path.abspath(src_enc) != os.path.abspath(dst_enc):
                shutil.move(src_enc, dst_enc)

            # Ensure only encrypted result remains in results/ (remove any stale plaintext CSV).
            dst_csv = os.path.join(RESULT_DIR, f"{result_stem}.csv")
            if os.path.exists(dst_csv):
                os.remove(dst_csv)

            # Remove intermediate per-video CSV artifacts from detect storage
            video_basename = os.path.basename(decrypted_file)
            intermediate_csv_paths = [
                os.path.join(BASE_DIR, f"{video_basename}.csv"),
                os.path.join(BASE_DIR, f"{video_basename}_movement.csv"),
                os.path.join(BASE_DIR, "_movement.csv"),
            ]
            for intermediate_csv in intermediate_csv_paths:
                if os.path.exists(intermediate_csv):
                    os.remove(intermediate_csv)

            set_job_state(job_id, status="done", finished_at=now_iso(), cv_duration_sec=cv_duration_sec)
            print(f"[video-worker] Video processing finished: {job_id} ({os.path.basename(decrypted_file)})")
            print(f"[video-worker] Saved encrypted result: {dst_enc}")
            print(f"[video-worker] Removed plaintext CSV if present: {dst_csv}")
        except Exception as exc:
            set_job_state(job_id, status="failed", finished_at=now_iso(), error=str(exc))
            print(f"[video-worker] Job failed: {job_id} - {exc}")
            traceback.print_exc()
        finally:
            video_job_queue.task_done()


@app.on_event("startup")
def startup_worker():
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(target=video_worker, name="video-worker", daemon=True)
        worker_thread.start()
        print("[video-worker] Startup completed, worker thread started")


@app.on_event("shutdown")
def shutdown_worker():
    if worker_thread is not None and worker_thread.is_alive():
        video_job_queue.put(STOP_SENTINEL)
        worker_thread.join(timeout=5)


def is_cv_running() -> bool:
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if "python" in (p.info.get("name") or "").lower() and "predict.py" in cmd:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


# -------- Routes --------

# Creates JWT token and returns to Android device
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=expires_delta)
    return {"access_token": access_token, "token_type": "bearer", "uid": user["UID"]}

# Revokes JWT token
@app.post("/revoke/")
async def revoke_token(token: str = Body(...)):
    revoked_tokens.add(token)
    return {"detail": "Token revoked successfully"}

# Server sends file to the Android device
@app.get("/download/{filename}")
async def download_file(filename: str, current_user: dict = Depends(get_current_user)):
    file_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

# Android device sends video to server
@app.post("/upload/")
@limiter.limit("2/5minute") # limited to 2 uploads per 5 minutes
async def upload_file(request: Request, file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    content = await file.read()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)
    print(f"User {current_user['username']} uploaded file: {file.filename}, size: {len(content)} bytes")
    output_name = os.path.splitext(file.filename)[0] + ".mp4"
    decrypted_file = os.path.join(UPLOAD_DIR, output_name)

    job_id = str(uuid4())
    with video_jobs_lock:
        video_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "filename": file.filename,
            "decrypted_filename": output_name,
            "result_stem": current_user.get("UID", current_user["username"]),
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "cv_duration_sec": None,
            "error": None,
        }

    video_job_queue.put({
        "job_id": job_id,
        "encrypted_file": file_path,
        "decrypted_file": decrypted_file,
        "username": current_user["username"],
        "user_uid": current_user.get("UID"),
    })
    print(f"[video-worker] Job queued: {job_id} (queue size: {video_job_queue.qsize()})")
    return JSONResponse(
        {
            "job_id": job_id,
            "filename": file.filename,
            "result_file": f"{current_user.get('UID', current_user['username'])}.enc",
            "message": "File uploaded. Processing started in background.",
        },
        status_code=202,
    )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, current_user: dict = Depends(get_current_user)):
    with video_jobs_lock:
        job = video_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

# Server health check
@app.get("/health")
def health():
    worker_alive = worker_thread is not None and worker_thread.is_alive()
    return {"status": "ok", "worker_alive": worker_alive, "queued_jobs": video_job_queue.qsize()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, ssl_certfile="cert.pem", ssl_keyfile="key.pem") # start server
