from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Header, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from datetime import datetime, timedelta
import uvicorn
import sys
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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
revoked_tokens = set()

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
    
def analyze_video():
    """Run predict.py on all of the videos from video folder"""
    if is_cv_running: # if video processing is already in progress, don't start again
        return
    predict_script_path = os.path.join(BASE_DIR, "predict.py")
    return  # currently exit, until server 
    predict_command = [
        "python",
        predict_script_path,
        "model=yolov8l.pt",
        "show=False",
        "source=video",
        "save=False",
        "save_txt=False"
    ]

    subprocess.run(predict_command)


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
    file_path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

# Android device sends video to server
@app.post("/upload/")
@limiter.limit("2/5minute") # limited to 2 uploads per 5 minutes
async def upload_file(request: Request, file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    content = await file.read()
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)
    print(f"User {current_user['username']} uploaded file: {file.filename}, size: {len(content)} bytes")
    output_file = file.filename.split(".")
    LEAdecryptCTR.decrypt_video(os.path.join(BASE_DIR, UPLOAD_FOLDER, file.filename), os.path.join(BASE_DIR, UPLOAD_FOLDER, output_file[0] + ".mp4"))
    os.remove(os.path.join(BASE_DIR, UPLOAD_FOLDER, file.filename))
    analyze_video()
    return JSONResponse({"filename": file.filename, "message": "File uploaded successfully!"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, ssl_certfile="cert.pem", ssl_keyfile="key.pem") # start server
