import os
import LEA  # Make sure the LEA library from the manual is installed/imported
import time
import pandas as pd

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
    leaCBC= LEA.CBC(LEA.ENCRYPT_MODE, key, iv)
    
    # Open input video and output encrypted file
    with open(input_file, "rb") as fin, open(output_file, "wb") as fout:
        data = fin.read()
        data_padded = pad(data)
        ct = leaCBC.update(data_padded)
        fout.write(ct)
        fout.write(leaCBC.final())

if __name__ == "__main__":
    start = time.time()
    encrypt_file("Bx3ixD3oj4gBxLp2GrxTLCYG3Wu2.csv", "Bx3ixD3oj4gBxLp2GrxTLCYG3Wu2.enc")
    df = pd.read_csv("Bx3ixD3oj4gBxLp2GrxTLCYG3Wu2.csv", encoding="latin1", on_bad_lines="skip")
    print(df)
    end = time.time()
    print("Encryption complete. Output file: Bx3ixD3oj4gBxLp2GrxTLCYG3Wu2.enc")
    print(end - start)