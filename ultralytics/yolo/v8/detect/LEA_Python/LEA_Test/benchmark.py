#-*- coding: utf-8 -*-
import time

import LEA


class benchmark:
    
    cpu_freq = 3.6
    
    def HiResTime(self):
        return time.time()
    
    def calibrate(self):
        t0 = 0
        t1 = 0
        dtMin = 0xffffffff
        
        for i in range(0, 100, 1):
            t0 = self.HiResTime()
            t1 = self.HiResTime()
            
            if(dtMin > t1 - t0):
                dtMin = t1 - t0
        
        return dtMin
    
    def get_cpb(self, time, data_len):
        return self.cpu_freq * (time / data_len) * 1000000000
    
    def lea_ecb_benchmark(self):
        tMin = 0xfffffff
        t0 = 0
        t1 = 0
        calibration = self.calibrate()

        mk = bytearray(32)
        src1 = bytearray(1000*16)
        src2 = bytearray(1000*16)
        outlen = [1000*16]
        print("***** lea_ecb_benchmark *****\n")
        print("- Encryption Speed\n")
        for key_len in (16,24,32):
            tMin = 0xfffffff
            
            for sample_count in range(0, 100, 1):
                lea = LEA.ECB(True,mk[:key_len])
                
                t0 = self.HiResTime()

                src1 = lea.update(src2)
                src2 = lea.update(src1)
                
                t1 = self.HiResTime()
                
                lea.final()
                
                if(tMin > t1 - t0 - calibration):
                    tMin = t1 - t0 - calibration
            
            print("[%d] %7.2f cycles/byte*\n" % (key_len * 8, self.get_cpb(tMin, 1000 * 16 * 2)))
        
        print("- Decryption Speed\n")
        for key_len in (16,24,32):
            tMin = 0xfffffff
            
            for sample_count in range(0, 100, 1):
                lea = LEA.ECB(True,mk[:key_len])
                
                t0 = self.HiResTime()
                
                src1 = lea.update(src2)
                src2 = lea.update(src1)
                
                t1 = self.HiResTime()
                
                lea.final()
                
                if(tMin > t1 - t0 - calibration):
                    tMin = t1 - t0 - calibration
            
            print("[%d] %7.2f cycles/byte*\n" % (key_len * 8, self.get_cpb(tMin, 1000 * 16 * 2)))