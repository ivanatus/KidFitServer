import os
#import LEA  # Make sure the LEA library from the manual is installed/imported
from LEA_Python.LEA import CBC, ENCRYPT_MODE
import time

BLOCK_SIZE = 16  # LEA block size (128 bits)

# Padding for CBC (PKCS7)
def pad(data):
    pad_len = BLOCK_SIZE - (len(data) % BLOCK_SIZE)
    return data + bytes([pad_len] * pad_len)

def encrypt_file(input_file, output_file):
    # 128-bit (16 byte) hardcoded key
    key = bytes([
        0x00, 0x11, 0x22, 0x33,
        0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xaa, 0xbb,
        0xcc, 0xdd, 0xee, 0xff
    ])
    
    # 16-byte IV (initial counter)
    iv = bytes([
        0x12, 0x34, 0x56, 0x78,
        0x90, 0xab, 0xcd, 0xef,
        0x01, 0x23, 0x45, 0x67,
        0x89, 0xab, 0xcd, 0xef
    ])
    
    # Create CBC cipher object
    leaCBC= CBC(ENCRYPT_MODE, key, iv)
    
    # Open input video and output encrypted file
    with open(input_file, "rb") as fin, open(output_file, "wb") as fout:
        data = fin.read()
        data_padded = pad(data)
        ct = leaCBC.update(data_padded)
        fout.write(ct)
        fout.write(leaCBC.final())

if __name__ == "__main__":
    start = time.time()
    encrypt_file("DataBase.csv", "DataBase.enc")
    end = time.time()
    print("Encryption complete. Output file: DataBase.enc")
    print(end - start)
