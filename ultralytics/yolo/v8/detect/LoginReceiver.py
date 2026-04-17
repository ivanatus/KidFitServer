from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from datetime import datetime, timedelta
import uvicorn
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "LEA_Python"))
import LEAdecrypt
import pandas as pd
import csv

# ---------------------------
# Config
# ---------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
API_TOKEN = "1001001"  # optional extra token header

SECRET_KEY = "my_jwt_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# Decrypt and get user database from csv
def get_user_database():
    LEAdecrypt.decrypt_file("DataBase.enc", "DataBase.csv")
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

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ---------------------------
# App
# ---------------------------
app = FastAPI()

# ---------------------------
# Utility functions
# ---------------------------
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    fake_users_db = get_user_database()
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
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

def verify_api_token(x_api_key: str = Header(...)):
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------------------
# Routes
# ---------------------------
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...),  token: str = Depends(verify_api_token), current_user: dict = Depends(get_current_user)):
    content = await file.read()
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)
    print(f"User {current_user['username']} uploaded file: {file.filename}, size: {len(content)} bytes")
    return JSONResponse({"filename": file.filename, "message": "File uploaded successfully!"})

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=6339, ssl_certfile="cert.pem", ssl_keyfile="key.pem", reload=True)
