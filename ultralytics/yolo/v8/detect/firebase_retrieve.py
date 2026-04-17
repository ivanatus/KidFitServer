import os
#provide path to globals.py file
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "deep_sort_pytorch", "deep_sort", "sort"))
from globals import Globals
sys.path.append(os.path.join(BASE_DIR, "LEA_Python"))
import LEAdecrypt
import csv
import subprocess

#Firebase setup
import firebase_admin
from firebase_admin import credentials
json_path = os.path.join(BASE_DIR, "newServiceAccountKey.json")
cred = credentials.Certificate(json_path)
#initialization of relevant Firebase services (storage and database)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://exercise-project-bc0dc-default-rtdb.europe-west1.firebasedatabase.app/',
    'storageBucket': 'exercise-project-bc0dc.appspot.com'
})
from firebase_admin import storage
from firebase_admin import db

def create_directory(directory_path):
    """Create "video" folder which will contain all videos downloaded from Firebase"""
    try:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except OSError as e:
        print(f"Error creating directory '{directory_path}': {e}")

def get_video_from_database():
    """Retrieve a video file from Firebase Cloud Storage."""
    bucket = storage.bucket() #reference to Firebase Storage
    folder_path = "Videos/New/"
    blobs = bucket.list_blobs(prefix=folder_path) #files in Firebase Storage

    found = False
    unanalyzed = True

    #has to be changed dependent on computer from which the code is being run
    video_directory = os.path.join(BASE_DIR, "video")

    #download all videos from Firebase Storage to video folder and delete them from Firebase
    for blob in blobs:    
        filename = os.path.basename(blob.name)
        print("Downloading file:", filename)

        #if undefined - delete from storage and continue to next
        if filename == "null":
            blob.delete()
            continue

        destination_file_path = os.path.join(video_directory, os.path.basename(blob.name))
        # Download the file
        blob.download_to_filename(destination_file_path)
        print("File exists ", os.path.exists(destination_file_path))
        print("File path ", destination_file_path)
        # Decrypt immediately after download
        filename = os.path.basename(blob.name)
        filename = filename.split(".")
        filename = filename[0] + ".mp4"
        print("Decrypting file: ", filename)
        LEAdecrypt.decrypt_video(destination_file_path, os.path.join(video_directory, filename))
        print("Decrypted file saved:", os.path.join(video_directory, filename))

        found = True
        print("Deleting file from Firebase:", os.path.basename(blob.name))
        blob.delete()

    #if there is no new videos in Firebase Storage, set flag to false
    if not found: 
        print("No new videos in database.")
        db_ref = db.reference('Unanalyzed')
        db_ref.set(False)
    
    call_yolo_script()

def call_yolo_script():
    """Run predict.py on all of the videos from video folder"""
    print("In call_yolo_script.")
    predict_script_path = os.path.join(BASE_DIR, "predict.py")
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

if __name__ == "__main__":
    # Specify the directory path - computer dependent
    directory_path = os.path.join(BASE_DIR, "video")
    create_directory(directory_path)
    get_video_from_database()