#-*- coding: utf-8 -*-
from .LEA import LEA
from .CipherMode import CipherMode

class CMAC(CipherMode):
    mac = None
    def gen_subkey(self):
        self.buffer = bytearray()
        K1 = bytearray(16)
        K2 = self.lea.encrypt(bytearray(16))

        if K2[0]&0x80 == 0x80:
            Rb = 0x87
        else:
            Rb = 0
        for i in range(15):
            K1[i] = ((K2[i] << 1) & 0xfe) ^ ((K2[i+1] >> 7) & 0x01)
        K1[15] = ((K2[15] << 1) & 0xfe ^ Rb)

        if K1[0]&0x80 == 0x80:
            Rb = 0x87
        else:
            Rb = 0

        for i in range(15):
            K2[i] = (((K1[i] << 1) & 0xfe) ^ ((K1[i+1] >> 7) & 0x01))
        K2[15] = ((K1[15] << 1) & 0xfe ^ Rb)

        return K1, K2

    def __init__(self, key):
        self.lea = LEA(key)

        self.K1, self.K2 = self.gen_subkey()
        self.update = self.generate
        self.mac = bytearray(16)

    def generate(self, data):
        if data is None:
            raise AttributeError('Improper data')
        if self.no_more:
            raise RuntimeError('Already finished')

        self.buffer += LEA.to_bytearray(data)
        offset = 0

        len_x16 = len(self.buffer)-16
        while offset < len_x16:
            self.mac = self.lea.encrypt(LEA.xorAr(self.mac, self.buffer[offset:offset+16]))
            offset += 16

        if offset != 0:
            self.buffer = self.buffer[offset:]

        return b''

    def final(self, mac=None):
        self.no_more = True
        if len(self.buffer) == 16:
            self.mac = LEA.xorAr(self.mac, self.buffer)
            self.mac = self.lea.encrypt(LEA.xorAr(self.K1, self.mac))
        else:
            self.mac = LEA.xorAr(self.mac, self.buffer)
            self.mac[len(self.buffer)] ^= 0x80
            self.mac = self.lea.encrypt(LEA.xorAr(self.K2, self.mac))

        if mac is not None:
            if bytearray(16) == LEA.xorAr(mac,self.mac):
                return mac
            else:
                return None
        return self.mac