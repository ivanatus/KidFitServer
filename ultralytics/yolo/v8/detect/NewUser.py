import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "LEA_Python"))
import LEAdecrypt
import LEAencrypt
import pandas as pd
import csv
import getpass

# Paths to encrypted and decrypted databases
csv_path = os.path.join(BASE_DIR, "DataBase.csv")
enc_path = os.path.join(BASE_DIR, "DataBase.enc")

# Permission only if admin pass
ADMIN_PASSWORD = "1234567890"
admin_password = getpass.getpass("Enter admin password: ")
if admin_password != ADMIN_PASSWORD:
    print("Access denied.")
    sys.exit(1)

# Decrypt file
LEAdecrypt.decrypt_file(enc_path, csv_path)

# Get all values already in database
df = pd.read_csv(csv_path, encoding="latin1", on_bad_lines="skip")
#print(df)
df.columns = [col.strip().lower() for col in df.columns]
df["password"] = df["password"].astype(str)
os.remove(csv_path)

# Enter values - username and UID have to be unique, nothing can be empty
while True:
    username = input("Enter username: ").strip()
    if username.lower() in df["username"].str.lower().values:
        print("Username already exists! Please enter a new one.")
    elif username == "":
        print("Username cannot be empty.")
    else:
        break

while True:
    password = input("Enter password: ").strip()
    if password == "":
        print("Password cannot be empty.")
    else:
        break

while True:
    uid = input("Enter UID: ").strip()
    if uid.lower() in df["uid"].str.lower().values:
        print("UID already exists! Please enter a new one.")
    elif uid == "":
        print("UID cannot be empty.")
    else:
        break

# Save values
df = df.astype(str)
df.loc[df.size + 1] = [str(username), str(password), str(uid)]
df.to_csv(csv_path, index=False, encoding="latin1")

os.remove(enc_path) # remove encrypted database after all values are properly added

#df = pd.read_csv(csv_path, encoding="latin1", on_bad_lines="skip")
#print(df)

# Encrypt database with new values and remove unencrypted database
LEAencrypt.encrypt_file(csv_path, enc_path)
os.remove(csv_path)
print("User successfully added!")