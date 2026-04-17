#-*- coding: utf-8 -*-
from .LEA import LEA
from .CipherMode import CipherMode
from .CTR import CTR
from .CipherMode import TagError

class CCM(CipherMode):
    ctr_mode = None
    nonce = None
    tag = None
    tag_len = None
    aad = None

    cbc_mac_len = None
    
    def keep_data(self, until_final):
        self.keep_data_until_final = until_final

    def __init__(self, do_enc, key, nonce, aad, tag_len, data_len):
        self.buffer = bytearray()
        self.nonce = LEA.to_bytearray(nonce, 'Nonce', forcecopy=True)
        if not 7 <= len(self.nonce) <= 13:
            raise AttributeError('Improper nonce length')
        if tag_len < 4 or tag_len > 16 or tag_len % 2 != 0:
            raise AttributeError('Invalid tag length')

        aad_len = 0

        self.tag_len = tag_len

        if aad is not None and len(aad) > 0:

            aad = LEA.to_bytearray(aad, 'aad')
            aad_len = len(aad)

            if aad_len < 0x1000 - 0x100:
                aad = bytearray([(aad_len >> 8) & 0xff,
                                 len(aad) & 0xff]) + aad
            elif aad_len >= (1 << 32):
                aad = bytearray([0xFF,
                                 0xFF,
                                 aad_len >> 56 & 0xff,
                                 aad_len >> 48 & 0xff,
                                 aad_len >> 40 & 0xff,
                                 aad_len >> 32 & 0xff,
                                 aad_len >> 24 & 0xff,
                                 aad_len >> 16 & 0xff,
                                 aad_len >> 8 & 0xff,
                                 aad_len & 0xff]) + aad
            else:
                aad = bytearray([0xFF,
                                 0xFE,
                                 aad_len >> 24 & 0xff,
                                 aad_len >> 16 & 0xff,
                                 aad_len >> 8 & 0xff,
                                 aad_len & 0xff]) + aad
        else:
            aad_len = 0
            aad = b''

        # En/decryption part
        ctr = bytearray([14-len(self.nonce)]) + self.nonce
        ctr += b'\x00'*ctr[0] + b'\x01'
        self.ctr_mode = CTR(True, key, ctr)
        self.lea = self.ctr_mode.lea  # for Initialization with CipherMode.lea as a key

        self.tag = bytearray([0])

        if aad_len != 0:
            self.tag[0] = 0x40 | (((tag_len - 2) >> 1) << 3) | (14 - len(nonce))
        else:
            self.tag[0] =    0 | (((tag_len - 2) >> 1) << 3) | (14 - len(nonce))

        self.tag += nonce
        self.tag += b'\x00' * (16-len(self.tag))

        # CBC_MAC part
        tmp_tag_val = data_len
        for i in range(15, len(self.nonce), -1):
            self.tag[i] = tmp_tag_val & 0xff
            tmp_tag_val >>= 8



        if tmp_tag_val > 0:
            raise AttributeError('Data is too long')

        self.tag = self.lea.encrypt(self.tag)
        offset = 0
        len_cbc_mac_x16 = len(aad) - 16
        if aad_len > 0:
            while offset <= len_cbc_mac_x16:
                self.tag = self.lea.encrypt(LEA.xorAr(self.tag, aad[offset:offset+16]))
                offset += 16

            if offset < len(aad):
                self.tag = self.lea.encrypt(LEA.xorAr(self.tag, aad[offset:]))
        if do_enc:
            self.update = self.encrypt
            self.keep_data_until_final = False
        else:
            self.update = self.decrypt
            self.outBuff = bytearray()
            self.keep_data_until_final = True

    def encrypt(self, pt):
        if pt is None:
            raise AttributeError('Improper pt')
        if self.no_more:
            raise RuntimeError('Already finished')

        offset = 0
        pt = LEA.to_bytearray(pt)
        self.buffer += pt
        ct = bytearray()
        len_x16 = len(self.buffer)-16
        while offset <= len_x16:
            ct += self.ctr_mode.update(self.buffer[offset:offset+16])
            self.tag = self.lea.encrypt(LEA.xorAr(self.tag, self.buffer[offset:offset+16]))
            offset += 16

        if offset != 0:
            self.buffer = self.buffer[offset:]

        return ct

    def decrypt(self, ct):
        if ct is None:
            raise AttributeError('Improper ct')
        if self.no_more:
            raise RuntimeError('Already finished')

        offset = 0
        ct = LEA.to_bytearray(ct)
        self.buffer += ct
        pt = bytearray()

        len_x16tag = len(self.buffer)-16-self.tag_len # last block and tag
        while offset <= len_x16tag:
            pt += self.ctr_mode.update(self.buffer[offset:offset+16])
            self.tag = self.lea.encrypt(LEA.xorAr(self.tag, pt[offset:offset+16]))

            offset += 16


        if offset != 0:
            self.buffer = self.buffer[offset:]
            self.outBuff += pt
            pt.zfill(len(pt))

        if self.keep_data_until_final:
            return bytearray()
        else:
            outBuff = self.outBuff
            self.outBuff = bytearray()
            return outBuff

    def final(self):
        if self.update == self.decrypt:
            if len(self.buffer) < self.tag_len:
                assert AttributeError('Invalid CT Length')
            tag = self.buffer[-self.tag_len:]
            self.buffer = self.buffer[:-self.tag_len]
        else:
            tag = None

        ctr_final = self.ctr_mode.update(self.buffer)
        ctr_final += self.ctr_mode.final()

        assert len(ctr_final) < 16

        self.no_more = True

        if len(self.buffer) > 0:
            if self.encrypt == self.update:
                self.tag = self.lea.encrypt(LEA.xorAr(self.tag, self.buffer))
            else:
                self.tag = self.lea.encrypt(LEA.xorAr(self.tag, ctr_final))

        ctr = self.ctr_mode.chain_vec
        for j in range(15- ctr[0], 16):
            ctr[j] = 0
        ctr = self.lea.encrypt(ctr)
        self.tag = LEA.xorAr(self.tag,ctr)[:self.tag_len]

        if self.update == self.decrypt:
            ctr_final = self.outBuff + ctr_final
            self.outBuff = None

        if tag is not None:
            if bytearray(self.tag_len) != LEA.xorAr(self.tag, tag)[:self.tag_len]:
                ctr_final.zfill(len(ctr_final))
                raise TagError('Invalid tag')

        if self.update == self.encrypt:
            return ctr_final + self.tag
        else:
            return ctr_final
