#-*- coding: utf-8 -*-
from .LEA import LEA
from .CipherMode import CipherMode

class OFB(CipherMode):
    chain_vec = b'\x00'*16

    def __init__(self, do_enc, key, iv):
        self.buffer = bytearray()
        self.lea = LEA(key)
        self.chain_vec = LEA.to_bytearray(iv, 'IV', forcecopy=True)

        self.encrypt = self.update
        self.decrypt = self.update

    def update(self, data):
        if data is None:
            raise AttributeError('Improper data length')
        if self.no_more:
            raise RuntimeError('Already finished')

        self.buffer += LEA.to_bytearray(data)
        offset = 0
        retVal = bytearray()

        len_x16 = len(self.buffer)-16
        while offset <= len_x16:
            self.chain_vec = self.lea.encrypt(self.chain_vec)
            retVal += LEA.xorAr(self.chain_vec, self.buffer[offset:offset+16])
            offset += 16

        if offset != 0:
            self.buffer = self.buffer[offset:]

        return retVal
            
    def final(self):
        self.no_more = True
        if len(self.buffer) > 0:
            return LEA.xorAr(self.lea.encrypt(self.chain_vec), self.buffer)[:len(self.buffer)]

        return b''