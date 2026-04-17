#-*- coding: utf-8 -*-
from .LEA import LEA
from .CipherMode import CipherMode

class CTR(CipherMode):
    chain_vec = b'\x00'*16
    use_ctr_len = 16
    def __init__(self, do_enc, key, ctr, use_ctr_len=16):
        if not 1 <= use_ctr_len <= 16:
            raise AttributeError('Improper use_len for CTR')
        self.buffer = bytearray()
        self.lea = LEA(key)
        self.chain_vec = LEA.to_bytearray(ctr, 'CTR', forcecopy=True)
        if len(self.chain_vec) < 16:
            raise AttributeError('Improper ctr')

        self.decrypt = self.update
        self.encrypt = self.update

        self.use_ctr_len = use_ctr_len

    def inc_ctr(self):
        for i in range(15, 15-self.use_ctr_len, -1):
            if self.chain_vec[i] != 0xff:
                self.chain_vec[i] += 1
                break
            else:
                self.chain_vec[i] = 0

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
            retVal += LEA.xorAr(self.lea.encrypt(self.chain_vec), self.buffer[offset:offset+16])
            self.inc_ctr()
            offset += 16

        if offset != 0:
            self.buffer = self.buffer[offset:]

        return retVal

    def final(self):
        self.no_more = True
        if len(self.buffer) > 0:
            return LEA.xorAr(self.lea.encrypt(self.chain_vec), self.buffer)[:len(self.buffer)]

        return b''