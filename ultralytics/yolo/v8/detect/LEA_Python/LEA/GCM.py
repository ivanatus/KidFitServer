#-*- coding: utf-8 -*-
from .LEA import LEA
from .CipherMode import CipherMode
from .CTR import CTR
from .CipherMode import TagError
import struct
class GCM(CipherMode):
    reduction_min = bytearray(
                    b"\x00\x00\x01\xc2\x03\x84\x02\x46\x07\x08\x06\xca\x04\x8c\x05\x4e" + \
                    b"\x0e\x10\x0f\xd2\x0d\x94\x0c\x56\x09\x18\x08\xda\x0a\x9c\x0b\x5e" + \
                    b"\x1c\x20\x1d\xe2\x1f\xa4\x1e\x66\x1b\x28\x1a\xea\x18\xac\x19\x6e" + \
                    b"\x12\x30\x13\xf2\x11\xb4\x10\x76\x15\x38\x14\xfa\x16\xbc\x17\x7e" + \
                    b"\x38\x40\x39\x82\x3b\xc4\x3a\x06\x3f\x48\x3e\x8a\x3c\xcc\x3d\x0e" + \
                    b"\x36\x50\x37\x92\x35\xd4\x34\x16\x31\x58\x30\x9a\x32\xdc\x33\x1e" + \
                    b"\x24\x60\x25\xa2\x27\xe4\x26\x26\x23\x68\x22\xaa\x20\xec\x21\x2e" + \
                    b"\x2a\x70\x2b\xb2\x29\xf4\x28\x36\x2d\x78\x2c\xba\x2e\xfc\x2f\x3e" + \
                    b"\x70\x80\x71\x42\x73\x04\x72\xc6\x77\x88\x76\x4a\x74\x0c\x75\xce" + \
                    b"\x7e\x90\x7f\x52\x7d\x14\x7c\xd6\x79\x98\x78\x5a\x7a\x1c\x7b\xde" + \
                    b"\x6c\xa0\x6d\x62\x6f\x24\x6e\xe6\x6b\xa8\x6a\x6a\x68\x2c\x69\xee" + \
                    b"\x62\xb0\x63\x72\x61\x34\x60\xf6\x65\xb8\x64\x7a\x66\x3c\x67\xfe" + \
                    b"\x48\xc0\x49\x02\x4b\x44\x4a\x86\x4f\xc8\x4e\x0a\x4c\x4c\x4d\x8e" + \
                    b"\x46\xd0\x47\x12\x45\x54\x44\x96\x41\xd8\x40\x1a\x42\x5c\x43\x9e" + \
                    b"\x54\xe0\x55\x22\x57\x64\x56\xa6\x53\xe8\x52\x2a\x50\x6c\x51\xae" + \
                    b"\x5a\xf0\x5b\x32\x59\x74\x58\xb6\x5d\xf8\x5c\x3a\x5e\x7c\x5f\xbe" + \
                    b"\xe1\x00\xe0\xc2\xe2\x84\xe3\x46\xe6\x08\xe7\xca\xe5\x8c\xe4\x4e" + \
                    b"\xef\x10\xee\xd2\xec\x94\xed\x56\xe8\x18\xe9\xda\xeb\x9c\xea\x5e" + \
                    b"\xfd\x20\xfc\xe2\xfe\xa4\xff\x66\xfa\x28\xfb\xea\xf9\xac\xf8\x6e" + \
                    b"\xf3\x30\xf2\xf2\xf0\xb4\xf1\x76\xf4\x38\xf5\xfa\xf7\xbc\xf6\x7e" + \
                    b"\xd9\x40\xd8\x82\xda\xc4\xdb\x06\xde\x48\xdf\x8a\xdd\xcc\xdc\x0e" + \
                    b"\xd7\x50\xd6\x92\xd4\xd4\xd5\x16\xd0\x58\xd1\x9a\xd3\xdc\xd2\x1e" + \
                    b"\xc5\x60\xc4\xa2\xc6\xe4\xc7\x26\xc2\x68\xc3\xaa\xc1\xec\xc0\x2e" + \
                    b"\xcb\x70\xca\xb2\xc8\xf4\xc9\x36\xcc\x78\xcd\xba\xcf\xfc\xce\x3e" + \
                    b"\x91\x80\x90\x42\x92\x04\x93\xc6\x96\x88\x97\x4a\x95\x0c\x94\xce" + \
                    b"\x9f\x90\x9e\x52\x9c\x14\x9d\xd6\x98\x98\x99\x5a\x9b\x1c\x9a\xde" + \
                    b"\x8d\xa0\x8c\x62\x8e\x24\x8f\xe6\x8a\xa8\x8b\x6a\x89\x2c\x88\xee" + \
                    b"\x83\xb0\x82\x72\x80\x34\x81\xf6\x84\xb8\x85\x7a\x87\x3c\x86\xfe" + \
                    b"\xa9\xc0\xa8\x02\xaa\x44\xab\x86\xae\xc8\xaf\x0a\xad\x4c\xac\x8e" + \
                    b"\xa7\xd0\xa6\x12\xa4\x54\xa5\x96\xa0\xd8\xa1\x1a\xa3\x5c\xa2\x9e" + \
                    b"\xb5\xe0\xb4\x22\xb6\x64\xb7\xa6\xb2\xe8\xb3\x2a\xb1\x6c\xb0\xae" + \
                    b"\xbb\xf0\xba\x32\xb8\x74\xb9\xb6\xbc\xf8\xbd\x3a\xbf\x7c\xbe\xbe")
    hTable = None

    ctr_mode = None
    tag = None

    msg_len = 0
    aad_len = 0
    tac_len = 0

    def init_8bit_table(self, h):
        hTable = bytearray(4096)
        temp = bytearray(16)

        for j in range(0, 16, 1):
            hTable[2048+j] = temp[j] = h[j] & 0xff

        j = 0x40
        while j >= 1:
            for k in range(15, 0, -1):
                temp[k] = (((temp[k] >> 1) & 0x7f) | ((temp[k - 1] << 7) & 0x80)) & 0xff
            temp[0] = ((temp[0] >> 1) & 0x7f) & 0xff

            if (hTable[32*j+15] & 0x01) != 0 :
                temp[0] = (temp[0] ^ 0xe1) & 0xff

            for k in range(0, 16, 1):
                hTable[16*j + k] = temp[k] & 0xff
            j >>= 1

        j = 2
        while j < 256:
            for k in range(1, j, 1):
                for l in range(0, 16, 1):
                    hTable[16*(j+k) + l] = (hTable[16*j + l] ^ hTable[16*k + l])&0xff
            j <<= 1

        self.hTable = hTable

    def gcm_ghash(self, r, data):
        offset = 0

        temp = bytearray(r)

        len_data_x16 = len(data) - 16
        while offset <= len_data_x16:
            temp = LEA.xorAr(temp,data[offset:offset+16])
            temp = self.gcm_gfmul(temp)
            offset += 16

        if offset < len(data):
            temp = LEA.xorAr(temp, data[offset:])
            temp = self.gcm_gfmul(temp)
        return temp

    def gcm_gfmul(self, x):
        temp = bytearray(16)

        for i in range(15, 0, -1):
            temp = LEA.xorAr(temp, self.hTable[16*x[i]:16*x[i] + 16])

            mask = temp.pop()
            temp.insert(0, 0)

            temp[0] ^= self.reduction_min[mask*2 + 0]
            temp[1] ^= self.reduction_min[mask*2 + 1]

        i = x[0]

        return LEA.xorAr(temp, self.hTable[16*i:16*i+16])
    
    def keep_data(self, until_final):
        self.keep_data_until_final = until_final
        
    def __init__(self, do_enc, key, nonce, aad, tag_len):
        self.buffer = bytearray()
        nonce = LEA.to_bytearray(nonce, 'nonce', forcecopy=True)

        aad = LEA.to_bytearray(aad, 'aad')
        self.aad_len = len(aad)
        self.tag_len = tag_len

        if tag_len <= 0 or tag_len > 16:
            raise AttributeError('Invalid tag length')

        if len(nonce) == 0:
            raise AttributeError('Invalid nonce length')

        self.lea = LEA(key)

        self.init_8bit_table(self.lea.encrypt(bytearray(16)))
        self.tag = self.gcm_ghash(bytearray(16), aad)

        if len(nonce) == 12:
            chain_vec = nonce + b'\x00\x00\x00\x01'
        else:
            t = 0
            offset = 0
            len_nonce_x16 = len(nonce)-16
            chain_vec = self.gcm_ghash(bytearray(16), nonce)

            len0 =  bytearray(struct.pack(">QQ", 0, (len(nonce)<<3)))

            chain_vec = self.gcm_ghash(chain_vec, len0)

        self.ctr_mode = CTR(True, self.lea, chain_vec, 4)
        self.lea = self.ctr_mode.lea
        self.E0 = self.lea.encrypt(chain_vec)
        self.ctr_mode.inc_ctr()

        if do_enc:
            self.update = self.encrypt
            self.keep_data_until_final = False
        else:
            self.update = self.decrypt
            self.outBuff = bytearray()
            self.keep_data_until_final = True

    def encrypt(self, pt):
        if pt is None:
            raise AttributeError('Improper pt length')
        if self.no_more:
            raise RuntimeError('Already finished')
        self.buffer += LEA.to_bytearray(pt)
        ct = bytearray()

        offset=0
        len_x16 = len(self.buffer)-16

        while offset <= len_x16:
            temp = self.ctr_mode.update(self.buffer[offset:offset+16])
            ct += temp
            self.tag = self.gcm_ghash(self.tag, temp)
            offset += 16

        if offset != 0:
            self.buffer = self.buffer[offset:]

        self.msg_len += len(pt)

        return ct

    def decrypt(self, ct):
        if ct is None:
            raise AttributeError('Improper ct length')
        if self.no_more:
            raise RuntimeError('Already finished')
        if len(ct) == 0:
            return bytearray()
        self.buffer += LEA.to_bytearray(ct)
        offset=0
        len_x16tag = len(self.buffer)-16-16  # last block and tag
        while offset <= len_x16tag:
            self.outBuff += self.ctr_mode.update(self.buffer[offset:offset+16])
            self.tag = self.gcm_ghash(self.tag, self.buffer[offset:offset+16])
            offset += 16

        if offset != 0:
            self.buffer = self.buffer[offset:]

        self.msg_len += len(ct)
        if self.keep_data_until_final:
            return bytearray()
        else:
            outBuff = self.outBuff
            self.outBuff = bytearray()
            return outBuff

    def final(self):
        tag = None

        if self.update == self.encrypt:
            ctr_final = bytearray()
        else:
            if len(self.buffer) < self.tag_len:
                assert AttributeError('Invalid CT Length')
            tag = self.buffer[-self.tag_len:]
            self.buffer = self.buffer[:-self.tag_len]
            ctr_final = self.outBuff
            self.outBuff = None
            self.msg_len -= self.tag_len

        ctr_final += self.ctr_mode.update(self.buffer)
        ctr_final += self.ctr_mode.final()

        if len(self.buffer) > 0:
            if self.encrypt == self.update:
                temp = ctr_final + bytearray(16-len(ctr_final))
                self.tag = self.gcm_ghash(self.tag, temp)
            else:
                temp = self.buffer + bytearray(16-len(self.buffer))
                self.tag = self.gcm_ghash(self.tag, temp)

        self.no_more = True
        self.aad_len *= 8
        self.msg_len *= 8
        temp = bytearray(struct.pack('>QQ',self.aad_len,self.msg_len))

        self.tag = self.gcm_ghash(self.tag, temp)
        self.tag = LEA.xorAr(self.tag, self.E0)[:self.tag_len]

        if tag is not None:
            if bytearray(self.tag_len) != LEA.xorAr(self.tag,tag):
                ctr_final.zfill(len(ctr_final))
                raise TagError('Invalid tag')

        if self.update == self.encrypt:
            return ctr_final + self.tag
        else:
            return ctr_final

