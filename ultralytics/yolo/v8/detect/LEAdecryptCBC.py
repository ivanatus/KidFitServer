import os
#import LEA  # Make sure the LEA library from the manual is installed/imported
from LEA_Python.LEA import CBC, DECRYPT_MODE
import time

def unpad(data):
    pad_len = data[-1]
    return data[:-pad_len]

def decrypt_file(input_file, output_file):
    # 128-bit (16 byte) hardcoded key — must be the same as used in encryption
    #key = b"1234567890abcdef"
    key = bytes([
        0x00, 0x11, 0x22, 0x33,
        0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xaa, 0xbb,
        0xcc, 0xdd, 0xee, 0xff
    ])

    # 16-byte IV (initial counter) — must match the one used in encryption
    #iv = b"abcdefghijklmnop"
    iv = bytes([
        0x12, 0x34, 0x56, 0x78,
        0x90, 0xab, 0xcd, 0xef,
        0x01, 0x23, 0x45, 0x67,
        0x89, 0xab, 0xcd, 0xef
    ])
    
    # Create CBC cipher object in decryption mode
    leaCBC = CBC(DECRYPT_MODE, key, iv)
    
    # Open encrypted input file and decrypted output file
    with open(input_file, "rb") as fin, open(output_file, "wb") as fout:
        ct = fin.read()
        pt_padded = leaCBC.update(ct) + leaCBC.final()
        pt = unpad(pt_padded)
        fout.write(pt)
