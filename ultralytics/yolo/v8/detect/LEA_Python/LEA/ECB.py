#-*- coding: utf-8 -*-
from .LEA import LEA
from .CipherMode import CipherMode

class ECB(CipherMode):
    def __init__(self, do_enc, key, PKCS5Padding=False):
        self.buffer = bytearray()
        self.lea = LEA(key)
        self.PKCS5Padding = PKCS5Padding
        if do_enc:
            self.update = self.encrypt
        else:
            self.update = self.decrypt

    def encrypt(self, pt):
        if pt is None:
            raise AttributeError('Improper pt')
        if self.no_more:
            raise RuntimeError('Already finished')

        self.buffer += LEA.to_bytearray(pt)
        offset = 0
        ct = bytearray()
        len_x16 = len(self.buffer)-16
        while offset <= len_x16:
            ct += self.lea.encrypt(self.buffer[offset:offset+16])
            offset += 16

        if offset != 0:
            self.buffer = self.buffer[offset:]

        return ct

    def decrypt(self, ct):
        if ct is None:
            raise AttributeError('Improper ct')
        if self.no_more:
            raise RuntimeError('Already finished')

        self.buffer += LEA.to_bytearray(ct)
        offset = 0
        pt = bytearray()
        len_x16 = len(self.buffer)-16
        if self.PKCS5Padding and len_x16 % 16 == 0:
            len_x16 -= 16
        while offset <= len_x16:
            pt += self.lea.decrypt(self.buffer[offset:offset+16])
            offset += 16

        if offset != 0:
            self.buffer = self.buffer[offset:]

        return pt

    def final(self):
        if self.no_more:
            raise RuntimeError('Already finished')

        result = bytearray()
        if self.PKCS5Padding and self.encrypt == self.update:
            more = 16 - len(self.buffer)
            self.buffer += bytearray([more])*more
            result += self.lea.encrypt(self.buffer)

        elif self.PKCS5Padding and self.decrypt == self.update:
            if len(self.buffer) != 16:
                raise ValueError('Improper data length')
            self.buffer = self.lea.decrypt(self.buffer)
            more = self.buffer[-1]
            for i in range(16-more, 15):
                if self.buffer[i] != more:
                    raise ValueError('Padding error')
            result += self.buffer[:16-more]

        elif len(self.buffer) > 0:
            raise ValueError('Improper data length')
        self.buffer = bytearray()
        self.no_more = True
        return result