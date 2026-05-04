from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Header, Request, Body, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.exceptions import RequestValidationError
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
import cv2
import re
import numpy as np
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

# App version
APK_PATH = "/root/KidFitServer/apk/app-release.apk"
LATEST_VERSION = {
    "versionCode": 2,
    "versionName": "1.1",
    "apkUrl": "https://82.165.249.157/apk",
    "forceUpdate": False,
    "releaseNotes": "Bug fixes and performance improvements."
}

# App setup
app = FastAPI()
UPLOAD_FOLDER = "video"
RESULT_FOLDER = "results"
CALIBRATION_FOLDER = "calibration"
UPLOAD_DIR = os.path.join(BASE_DIR, UPLOAD_FOLDER)
RESULT_DIR = os.path.join(BASE_DIR, RESULT_FOLDER)
CALIBRATION_DIR = os.path.join(BASE_DIR, CALIBRATION_FOLDER)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CALIBRATION_DIR, exist_ok=True)
revoked_tokens = set()
video_job_queue = queue.Queue()
video_jobs = {}
video_jobs_lock = threading.Lock()
worker_thread = None
STOP_SENTINEL = object()
user_db_cache = None
user_db_cache_ts = None
user_db_lock = threading.Lock()
USER_DB_CACHE_TTL_SEC = 300
# CV runtime knobs (change these to trade speed vs accuracy)
CV_MODEL = "yolov8l.pt"
CV_IMGSZ = 512
CPU_TORCH_THREADS = max(1, (os.cpu_count() or 2) - 1)
CPU_OPENCV_THREADS = max(1, min(4, os.cpu_count() or 1))
MIN_CALIB_SHARPNESS_VAR = 70.0
MAX_CALIB_GLARE_RATIO = 0.12
MIN_A4_AREA_RATIO = 0.015
MAX_A4_AREA_RATIO = 0.95
MIN_A4_ASPECT = 1.15
MAX_A4_ASPECT = 1.70

# Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Print concise diagnostics to quickly identify which field/key is wrong in multipart requests.
    print("[validation] 422 RequestValidationError")
    print(f"[validation] path={request.url.path} method={request.method}")
    print(f"[validation] content-type={request.headers.get('content-type', '')}")
    print(f"[validation] errors={exc.errors()}")
    try:
        if "multipart/form-data" in (request.headers.get("content-type", "")):
            form = await request.form()
            form_keys = list(form.keys())
            print(f"[validation] multipart keys={form_keys}")
    except Exception as form_exc:
        print(f"[validation] could not inspect form keys: {form_exc}")

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )
    
# Decrypt and get user database from csv
def get_user_database():
    global user_db_cache, user_db_cache_ts
    now = time.time()
    with user_db_lock:
        if user_db_cache is not None and user_db_cache_ts is not None and (now - user_db_cache_ts) < USER_DB_CACHE_TTL_SEC:
            return user_db_cache

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
        user_db_cache = fake_users_db
        user_db_cache_ts = now
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


def extract_movement_value(stdout: str):
    if not stdout:
        return None
    matches = re.findall(r"Movement in this video:\s*([-+]?\d*\.?\d+)", stdout)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def analyze_video(app_version: str, calibration_is_calibrated: bool, calibration_saved_at: str, calibration_json: str):
    """Run predict.py on all of the videos from video folder"""
    predict_script_path = os.path.join(BASE_DIR, "predict.py")
    predict_command = [
        "/root/KidFitServer/venv/bin/python",
        predict_script_path,
        f"model={CV_MODEL}",
        f"imgsz={CV_IMGSZ}",
        "cls=0",
        "max_det=90",
        "verbose=False",
        "show=False",
        "source=video",
        "save=False",
        "save_txt=False"
    ]
    print("Start analysis")
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["APP_VERSION"] = app_version
    env["CALIBRATION_IS_CALIBRATED"] = "true" if calibration_is_calibrated else "false"
    env["CALIBRATION_SAVED_AT"] = str(calibration_saved_at or "")
    env["CALIBRATION_JSON"] = calibration_json or ""
    return subprocess.run(predict_command, cwd=BASE_DIR, capture_output=True, text=True, env=env)


def remux_video_for_stable_decode(video_path: str) -> bool:
    """Remux MP4 stream metadata to reduce FFmpeg/OpenCV frame retrieval errors."""
    temp_path = video_path + ".remux.mp4"
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("[video-worker] ffmpeg not found, falling back to OpenCV re-encode")
        return reencode_video_with_opencv(video_path)

    cmd = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-c",
        "copy",
        temp_path,
    ]
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            os.replace(temp_path, video_path)
            print(f"[video-worker] Video remux successful: {video_path}")
            return True
        print(f"[video-worker] Video remux failed (returncode={result.returncode}): {video_path}")
        if result.stderr:
            print(result.stderr, end="")
    except Exception as exc:
        print(f"[video-worker] Video remux exception: {exc}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return False


def reencode_video_with_opencv(video_path: str) -> bool:
    """Fallback path when ffmpeg is unavailable: decode and re-encode frames with OpenCV."""
    temp_path = video_path + ".reenc.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video-worker] OpenCV re-encode failed: cannot open {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        width, height = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"[video-worker] OpenCV re-encode failed: cannot create writer for {temp_path}")
        return False

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    try:
        if written > 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            os.replace(temp_path, video_path)
            print(f"[video-worker] OpenCV re-encode successful: {video_path} (frames: {written})")
            return True
        print(f"[video-worker] OpenCV re-encode produced no frames: {video_path}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return False


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Return 4 points ordered as: top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left
    return ordered


def _detect_a4_corners(image: np.ndarray) -> np.ndarray | None:
    """Detect the largest 4-corner contour, assumed to be A4 paper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = image.shape[:2]
    min_area = MIN_A4_AREA_RATIO * float(h * w)
    best = None
    best_area = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        if area > best_area:
            best_area = area
            best = approx

    if best is None:
        return None

    ordered = _order_quad_points(best.reshape(4, 2))
    # Refine contour corners for better reprojection quality.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(
        gray,
        ordered.reshape(-1, 1, 2).astype(np.float32),
        (7, 7),
        (-1, -1),
        criteria,
    )
    return refined.reshape(4, 2)


def _frame_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _glare_ratio(gray: np.ndarray) -> float:
    # Saturated highlights can break edge/quad detection.
    return float(np.mean(gray >= 245))


def _quad_area_ratio(quad: np.ndarray, img_w: int, img_h: int) -> float:
    area = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.float32))
    return float(area / max(1.0, float(img_w * img_h)))


def _quad_aspect_ratio(quad: np.ndarray) -> float:
    tl, tr, br, bl = quad
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    w = max(1e-6, (top + bottom) / 2.0)
    h = max(1e-6, (left + right) / 2.0)
    ratio = max(w, h) / min(w, h)
    return float(ratio)


def calibrate_camera_from_a4_images(
    image_paths: list[str],
    a4_width_m: float = 0.297,
    a4_height_m: float = 0.210,
) -> dict:
    """
    Calibrate camera from multiple views of a single A4 sheet.
    Returns RMS reprojection error, camera matrix and distortion coefficients.
    """
    if not image_paths:
        raise ValueError("No image paths provided for calibration.")

    # 3D corners of A4 on Z=0 plane (meters): TL, TR, BR, BL
    objp = np.array(
        [
            [0.0, 0.0, 0.0],
            [a4_width_m, 0.0, 0.0],
            [a4_width_m, a4_height_m, 0.0],
            [0.0, a4_height_m, 0.0],
        ],
        dtype=np.float32,
    )

    object_points = []
    image_points = []
    used_images = []
    rejected_images = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            rejected_images.append({"path": path, "reason": "unable_to_read"})
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])  # (width, height)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = _frame_sharpness(gray)
        if sharpness < MIN_CALIB_SHARPNESS_VAR:
            rejected_images.append(
                {"path": path, "reason": f"blurred_frame:lap_var={sharpness:.2f}"}
            )
            continue

        glare = _glare_ratio(gray)
        if glare > MAX_CALIB_GLARE_RATIO:
            rejected_images.append(
                {"path": path, "reason": f"glare_too_high:ratio={glare:.4f}"}
            )
            continue

        corners = _detect_a4_corners(img)
        if corners is None:
            rejected_images.append({"path": path, "reason": "a4_not_detected"})
            continue

        area_ratio = _quad_area_ratio(corners, img.shape[1], img.shape[0])
        if area_ratio < MIN_A4_AREA_RATIO or area_ratio > MAX_A4_AREA_RATIO:
            rejected_images.append(
                {"path": path, "reason": f"partial_or_invalid_area:ratio={area_ratio:.4f}"}
            )
            continue

        aspect = _quad_aspect_ratio(corners)
        if aspect < MIN_A4_ASPECT or aspect > MAX_A4_ASPECT:
            rejected_images.append(
                {"path": path, "reason": f"bad_a4_aspect:{aspect:.4f}"}
            )
            continue

        object_points.append(objp)
        image_points.append(corners.reshape(-1, 1, 2).astype(np.float32))
        used_images.append(path)

    if len(used_images) < 3:
        print(
            f"[calibration] Failed: not enough valid images. "
            f"total={len(image_paths)}, used={len(used_images)}, rejected={len(rejected_images)}"
        )
        if rejected_images:
            print(f"[calibration] Rejected details: {rejected_images}")
        raise ValueError(
            f"Not enough valid images for calibration. Need >= 3, got {len(used_images)}."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    print(
        f"[calibration] Frames summary: total={len(image_paths)}, "
        f"used={len(used_images)}, rejected={len(rejected_images)}, rms={float(rms):.4f}"
    )

    return {
        "rms_reprojection_error": float(rms),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
        "used_images_count": len(used_images),
        "used_images": used_images,
        "rejected_images_count": len(rejected_images),
        "rejected_images": rejected_images,
        "a4_size_m": {"width": a4_width_m, "height": a4_height_m},
    }


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
        app_version = job.get("app_version", "unknown")
        calibration_is_calibrated = bool(job.get("calibration_is_calibrated", False))
        calibration_saved_at = str(job.get("calibration_saved_at", ""))
        calibration_json = job.get("calibration_json", "")
        job_start = time.perf_counter()
        set_job_state(job_id, status="running", started_at=now_iso())
        print(f"[video-worker] Job started: {job_id} ({os.path.basename(encrypted_file)})")

        try:
            LEAdecryptCTR.decrypt_video(encrypted_file, decrypted_file)
            if os.path.exists(encrypted_file):
                os.remove(encrypted_file)
            remux_video_for_stable_decode(decrypted_file)

            cv_start = time.perf_counter()
            result = analyze_video(
                app_version,
                calibration_is_calibrated,
                calibration_saved_at,
                calibration_json,
            )
            cv_duration_sec = round(time.perf_counter() - cv_start, 3)
            print(f"[video-worker] CV processing time: {cv_duration_sec}s (job: {job_id})")
            if result.stdout:
                calibration_lines = [
                    line for line in result.stdout.splitlines()
                    if "[predict] Calibration" in line
                ]
                for line in calibration_lines:
                    print(line)
            movement_value = extract_movement_value(result.stdout or "")
            if result.returncode != 0:
                error_msg = (result.stderr or result.stdout or "predict.py failed").strip()
                if result.stderr:
                    print(result.stderr, end="")
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

            total_duration_sec = round(time.perf_counter() - job_start, 3)
            set_job_state(
                job_id,
                status="done",
                finished_at=now_iso(),
                cv_duration_sec=cv_duration_sec,
                total_duration_sec=total_duration_sec,
                ending_movement=movement_value,
            )
            print(f"[video-worker] Video processing finished: {job_id} ({os.path.basename(decrypted_file)})")
            print(f"[video-worker] Saved encrypted result: {dst_enc}")
            print(f"[video-worker] Removed plaintext CSV if present: {dst_csv}")
            print(f"[video-worker] Total processing time: {total_duration_sec}s (job: {job_id})")
            if movement_value is not None:
                print(f"[video-worker] Ending movement calculation: {movement_value}")
            else:
                print("[video-worker] Ending movement calculation: not found in predict.py output")
        except Exception as exc:
            set_job_state(job_id, status="failed", finished_at=now_iso(), error=str(exc))
            print(f"[video-worker] Job failed: {job_id} - {exc}")
            traceback.print_exc()
        finally:
            video_job_queue.task_done()


@app.on_event("startup")
def startup_worker():
    global worker_thread
    try:
        import torch
        torch.set_num_threads(CPU_TORCH_THREADS)
        print(f"[video-worker] Torch threads set to: {CPU_TORCH_THREADS}")
    except Exception as exc:
        print(f"[video-worker] Torch thread tuning skipped: {exc}")
    try:
        cv2.setNumThreads(CPU_OPENCV_THREADS)
        print(f"[video-worker] OpenCV threads set to: {CPU_OPENCV_THREADS}")
    except Exception as exc:
        print(f"[video-worker] OpenCV thread tuning skipped: {exc}")
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
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    calibration_is_calibrated: str | None = Form(None),
    calibration_saved_at: str | None = Form(None),
    calibration_json: str | None = Form(None),
    app_version: str | None = Header(None),
    current_user: dict = Depends(get_current_user)
):
    content = await file.read()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)
    print(f"User {current_user['username']} uploaded file: {file.filename}, size: {len(content)} bytes")
    output_name = os.path.splitext(file.filename)[0] + ".mp4"
    decrypted_file = os.path.join(UPLOAD_DIR, output_name)
    calibration_enabled = str(calibration_is_calibrated).strip().lower() == "true" if calibration_is_calibrated is not None else False

    job_id = str(uuid4())
    job_app_version = app_version if app_version else "unknown"
    with video_jobs_lock:
        video_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "filename": file.filename,
            "decrypted_filename": output_name,
            "result_stem": current_user.get("UID", current_user["username"]),
            "app_version": job_app_version,
            "calibration_is_calibrated": calibration_enabled,
            "calibration_saved_at": calibration_saved_at if calibration_saved_at is not None else "",
            "calibration_json": calibration_json if calibration_json is not None else "",
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
        "app_version": job_app_version,
        "calibration_is_calibrated": calibration_enabled,
        "calibration_saved_at": calibration_saved_at if calibration_saved_at is not None else "",
        "calibration_json": calibration_json if calibration_json is not None else "",
    })
    print(
        "[upload] calibration payload: "
        f"is_calibrated={calibration_enabled}, "
        f"saved_at={calibration_saved_at if calibration_saved_at is not None else ''}, "
        f"json_len={len(calibration_json) if calibration_json else 0}"
    )
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
def health(app_version: str | None = Header(None), current_user: dict = Depends(get_current_user)):
    worker_alive = worker_thread is not None and worker_thread.is_alive()
    latest_version = LATEST_VERSION.get("versionName")
    if app_version == latest_version:
        return {"status": "ok", "worker_alive": worker_alive, "queued_jobs": video_job_queue.qsize()}
    else:
        return JSONResponse({"message": "Update needed!"}, status_code=300)
    
# App update
@app.get("/apk")
def download_apk(current_user: dict = Depends(get_current_user)):
    print("APK call in progress.")
    return FileResponse(
        APK_PATH,
        media_type="application/vnd.android.package-archive",
        filename="app-release.apk"
    )


@app.post("/calibration/")
@limiter.limit("10/5minute")
async def upload_calibration_bundle(
    request: Request,
    uid: str = Form(...),
    files: list[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    token_uid = str(current_user.get("UID", ""))
    if not token_uid:
        raise HTTPException(status_code=401, detail="UID missing in token")
    if uid != token_uid:
        raise HTTPException(status_code=403, detail="UID does not match authenticated user")
    if not files:
        raise HTTPException(status_code=400, detail="No calibration files provided")

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_dir = os.path.join(CALIBRATION_DIR, uid, batch_id)
    os.makedirs(save_dir, exist_ok=True)

    saved_files = []
    for i, upload in enumerate(files, start=1):
        original_name = os.path.basename(upload.filename or f"calibration_{i}.jpg")
        if not original_name:
            original_name = f"calibration_{i}.jpg"
        file_path = os.path.join(save_dir, original_name)

        content = await upload.read()
        if len(content) == 0:
            continue
        with open(file_path, "wb") as f:
            f.write(content)
        saved_files.append(file_path)

    if not saved_files:
        raise HTTPException(status_code=400, detail="All uploaded calibration files were empty")

    calibration = None
    calibration_error = None
    try:
        calibration = calibrate_camera_from_a4_images(saved_files)
    except ValueError as exc:
        print(f"[calibration] ValueError during calibration: {exc}")
        print(f"[calibration] Input files count={len(saved_files)}")
        print(f"[calibration] Input files={saved_files}")
        calibration_error = HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        print(f"[calibration] Unexpected exception type={type(exc).__name__}: {exc}")
        print(f"[calibration] Input files count={len(saved_files)}")
        print(f"[calibration] Input files={saved_files}")
        calibration_error = HTTPException(status_code=500, detail=f"Calibration failed: {exc}")
    finally:
        # Uploaded calibration bundles are temporary; remove all files/folders after processing.
        try:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir, ignore_errors=True)
            uid_dir = os.path.join(CALIBRATION_DIR, uid)
            if os.path.isdir(uid_dir) and not os.listdir(uid_dir):
                os.rmdir(uid_dir)
        except Exception as cleanup_exc:
            print(f"[calibration] Cleanup warning for UID {uid}: {cleanup_exc}")

    if calibration_error is not None:
        print(
            f"[calibration] Returning error status={calibration_error.status_code}, "
            f"detail={calibration_error.detail}"
        )
        raise calibration_error

    used_count = calibration.get("used_images_count", 0) if isinstance(calibration, dict) else 0
    rejected_count = calibration.get("rejected_images_count", 0) if isinstance(calibration, dict) else 0
    print(f"[calibration] UID {uid} uploaded {len(saved_files)} files")
    print(f"[calibration] Paper detected/used: {used_count}, rejected: {rejected_count}")
    return JSONResponse(
        {
            "message": "Calibration completed.",
            "uid": uid,
            "batch_id": batch_id,
            "calibration": calibration,
        },
        status_code=201,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, ssl_certfile="cert.pem", ssl_keyfile="key.pem") # start server
