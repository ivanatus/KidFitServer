#-*- coding: utf-8 -*-
from .LEA import LEA
from .CipherMode import CipherMode

class CFB(CipherMode):
    chain_vec = b'\x00'*16

    def __init__(self, do_enc, key, iv):
        self.buffer = bytearray()
        self.lea = LEA(key)
        self.chain_vec = LEA.to_bytearray(iv, 'IV', forcecopy=True)

        if do_enc:
            self.update = self.encrypt
        else:
            self.update = self.decrypt

    def encrypt(self, pt):
        if pt is None:
            raise AttributeError('Improper pt length')
        if self.no_more:
            raise RuntimeError('Already finished')

        self.buffer += LEA.to_bytearray(pt)
        offset = 0
        ct = bytearray()

        len_x16 = len(self.buffer)-16
        while offset <= len_x16:
            self.chain_vec = LEA.xorAr(self.lea.encrypt(self.chain_vec), self.buffer[offset:offset+16])
            ct += self.chain_vec
            offset += 16

        if offset != 0:
            self.buffer = self.buffer[offset:]

        return ct

    def decrypt(self, ct):
        if ct is None:
            raise AttributeError('Improper pt length')
        if self.no_more:
            raise RuntimeError('Already finished')

        self.buffer += LEA.to_bytearray(ct)
        offset = 0
        pt = bytearray()

        len_x16 = len(self.buffer)-16
        while offset <= len_x16:
            temp = self.lea.encrypt(self.chain_vec)
            self.chain_vec = self.buffer[offset:offset+16]
            pt += LEA.xorAr(temp, self.chain_vec)

            offset += 16

        if offset != 0:
            self.buffer = self.buffer[offset:]

        return pt

    def final(self):
        self.no_more=True
        return LEA.xorAr(self.lea.encrypt(self.chain_vec), self.buffer)[:len(self.buffer)]

