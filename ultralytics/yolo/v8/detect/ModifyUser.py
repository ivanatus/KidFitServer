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
df.columns = [col.strip().lower() for col in df.columns]
df["password"] = df["password"].astype(str)
print(df)
os.remove(csv_path)

# Enter values of user to be modified - username or uid needed 
choice = ""
while choice not in ["1", "2"]:
    print("Find user by:")
    print("1 - Username")
    print("2 - UID")
    choice = input("Enter choice: ").strip()

if choice == "1":
    # Search by username
    while True:
        username = input("Enter username: ").strip()
        if username.lower() not in df["username"].str.lower().values:
            print("Username not found.")
        elif username == "":
            print("Username cannot be empty.")
        else:
            user_row = df[df["username"].str.lower() == username.lower()]
            break
else:
    # Search by UID
    while True:
        uid = input("Enter UID: ").strip()
        if uid.lower() not in df["uid"].str.lower().values:
            print("UID not found.")
        elif uid == "":
            print("UID cannot be empty.")
        else:
            user_row = df[df["uid"].str.lower() == uid.lower()]
            break

# Get current values
old_username = user_row["username"].values[0]
old_password = user_row["password"].values[0]
old_uid = user_row["uid"].values[0]

# New values - username, password, uid in order
while True:
    new_username = input(f"New username (Enter to keep '{old_username}'): ").strip()
    if new_username == "":
        print("Keeping the username.")
        new_username = old_username
        break
    elif new_username in df and new_username != old_username:
        print("Username already in use, enter another one!")
    else:
        break

while True:
    new_password = input(f"New password (Enter to keep old one).").strip()
    if new_password == "":
        print("Keeping the password.")
        new_password = old_password
        break
    else:
        break

while True:
    new_uid = input(f"New uid (Enter to keep '{old_uid}'): ").strip()
    if new_uid == "":
        print("Keeping the UID.")
        new_uid = old_uid
        break
    elif new_uid in df and new_uid != old_uid:
        print("UID already in use, enter another one!")
    else:
        break

# Locate row and change its values
df.loc[(df["username"].str.lower() == old_username.lower()) & (df["uid"].str.lower() == old_uid.lower()), ["username", "password", "uid"]] = [new_username, new_password, new_uid]

# Save new values
df = df.astype(str)
df.to_csv(csv_path, index=False, encoding="latin1")

os.remove(enc_path) # remove encrypted database after all values are properly added
print(df)
# Encrypt database with new values and remove unencrypted database
LEAencrypt.encrypt_file(csv_path, enc_path)
os.remove(csv_path)
print("User successfully modified!")