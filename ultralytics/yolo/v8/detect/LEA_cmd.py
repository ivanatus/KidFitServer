#-*- coding: utf-8 -*-
import argparse

import LEA_Test
import LEA


def printTest(result):
    if result is not None and type(result) is tuple and result[0] != 0:
        print(result)

def test():
    leaVS = LEA_Test.LEAvs()

    printTest(leaVS.lea_mmt_ecb_test())
    printTest(leaVS.lea_mmt_cbc_test())
    printTest(leaVS.lea_mmt_ctr_test())
    printTest(leaVS.lea_mmt_cfb_test())
    printTest(leaVS.lea_mmt_ofb_test())

    printTest(leaVS.lea_cmac_g_test())

    printTest(leaVS.lea_ccm_ge_test())

    printTest(leaVS.lea_gcm_ae_test())
    LEA_Test.benchmark().lea_ecb_benchmark()

def makeBinary(filename,length, fillValue = None):
    import random
    import struct
    buffer = bytearray()
    with open(filename,'wb') as fp:
        for idx in range(length):
            if fillValue is not None:
                buffer.append(fillValue)
            else:
                buffer.append(random.randint(0,255))

            if len(buffer) > 4096:
                fp.write(buffer)
                buffer = bytearray()

        fp.write(buffer)

def doEnc(env):
    key = env.key.read()
    data = env.input
    data_len = 0
    iv = None

    if env.mode != 'ECB':
        if env.iv is None:
            raise AttributeError('IV is required in mode.'%env.mode)
        else:
            iv = env.iv.read()

    if env.mode == 'ECB':
        leaEnc = LEA.ECB(env.enc, key)
    elif env.mode == 'CBC':
        leaEnc = LEA.CBC(env.enc, key, iv)
    elif env.mode == 'CTR':
        leaEnc = LEA.CTR(env.enc, key, iv)
    elif env.mode == 'CFB':
        leaEnc = LEA.CFB(env.enc, key, iv)
    elif env.mode == 'OFB':
        leaEnc = LEA.OFB(env.enc, key, iv)
    else:
        leaEnc = None
        raise AttributeError('Unknown Mode')

    while True:
        buffer = data.read(4096)
        if len(buffer) == 0:
            break
        data_len += len(buffer)

        env.output.write(leaEnc.update(buffer))
    env.output.write(leaEnc.final())

    if env.enc:
        enc_txt = 'Enc.'
    else:
        enc_txt = 'Dec.'

    #print('LEA %s %s - Done'%(env.mode,enc_txt))
    return True

def doAE(env):

    key = env.key.read()
    iv = env.iv.read()

    if env.aad is None:
        aad = b''
    else:
        aad = env.aad.read()

    tag_len = env.taglen

    if tag_len > 16 or tag_len <= 0:
        raise AttributeError('Invalid Tag Length')
    data = env.input

    data_len = 0

    try:
        if env.mode == 'CCM' and (LEA.LEA.py_under3 and not isinstance(data, file)):
            buffer = data.read()
            leaCCM = LEA.CCM(env.enc, key, iv, aad, tag_len, len(buffer))

            env.output.write(leaCCM.update(buffer))
            env.output.write(leaCCM.final())
        else:
            if env.mode == 'GCM':
                leaAE = LEA.GCM(env.enc, key, iv, aad, tag_len)
            else:
                data.seek(0,2)
                data_len = data.tell() - tag_len
                data.seek(0,0)
                leaAE = LEA.CCM(env.enc, key, iv, aad, tag_len, data_len)

            while True:
                buffer = data.read(4096)
                if len(buffer) == 0:
                    break
                data_len += len(buffer)
                env.output.write(leaAE.update(buffer))
            env.output.write(leaAE.final())
    except LEA.TagError:
        print('LEA %s Dec. - Tag value is inconsistent.' % env.mode)
        return False

    return True

def doCMAC(env):

    key = env.key.read()
    data = env.input
    leaCMAC = LEA.CMAC(key)
    data_len = 0

    while True:
        buffer = data.read(4096)
        if len(buffer) == 0:
            break
        data_len += len(buffer)

        leaCMAC.update(buffer)
    env.output.write(leaCMAC.final())
    return True
    #print('LEA CMAC - Done')

def buildParser():
    import sys
    import os
    basename = os.path.basename(sys.argv[0])
    parser = argparse.ArgumentParser(description='LEA for Python command line tool',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='Example: %s enc --help'%basename)
    subparsers = parser.add_subparsers()
    parser_enc = subparsers.add_parser('enc', description='Operatie en/decrpyt with LEA',
                                       help='Operatie en/decrpyt with LEA',
                                       epilog='Example: %s enc -e -m CTR -k key.bin --iv iv.bin -i pt.bin -o ct.bin'%basename)
    parser_enc_e = parser_enc.add_mutually_exclusive_group(required=True)
    parser_enc_e.add_argument('-e', '--encrypt', dest='enc', action='store_true', help='Encrypt file')
    parser_enc_e.add_argument('-d', '--decrypt', dest='enc', action='store_false', help='Decrypt file')
    parser_enc.add_argument('-m' , '--mode', type=str.upper, choices=['ECB','CBC','CTR','CFB','OFB'],
                            help='Mode of operation', required=True)
    parser_enc.add_argument('-k','--key', type=argparse.FileType('rb'), help='Key file(binary)', required=True)
    parser_enc.add_argument('--iv', '--nonce', type=argparse.FileType('rb'), help='IV file')
    parser_enc.add_argument('-i', '--input', type=argparse.FileType('rb'), help='Input file(binary)', required=True)
    parser_enc.add_argument('-o', '--output', type=argparse.FileType('wb'), help='Output file(binary)', required=True)
    parser_enc.set_defaults(func=doEnc)

    parser_ae = subparsers.add_parser('ae', description='Operate authenticated encryption with LEA',
                                      help='Operate authenticated encryption with LEA',
                                      epilog='Example: %s ae -e -m GCM -k key.bin --iv iv.bin --tag tag.bin -i pt.bin -o ct.bin'%basename)
    parser_ae_e = parser_ae.add_mutually_exclusive_group(required=True)
    parser_ae_e.add_argument('-e', '--encrypt', dest='enc', action='store_true', help='Encrypt file')
    parser_ae_e.add_argument('-d', '--decrypt', dest='enc', action='store_false', help='Decrypt file')
    parser_ae.add_argument('-m', '--mode', choices=['CCM','GCM'], type=str.upper, help ='Mode of operation',
                           required=True)
    parser_ae.add_argument('-k','--key', type=argparse.FileType('rb'), help='Key file(binary)', required=True)
    parser_ae.add_argument('--iv', '--nonce', type=argparse.FileType('rb'), help='IV file', required= True)
    parser_ae.add_argument('--aad', type=argparse.FileType('rb'), help='Additional authenticated data file(binary)')
    parser_ae.add_argument('--taglen', type=int, default=16, help='Tag length. should be equal or less than 16.')
    parser_ae.add_argument('-i', '--input', type=argparse.FileType('rb'), help='Input file(binary)', required=True)
    parser_ae.add_argument('-o,', '--output', type=argparse.FileType('wb'), help='Output file(binary)', required= True)
    parser_ae.set_defaults(func=doAE)

    parser_cmac = subparsers.add_parser('cmac', help='operation CMAC with LEA',
                                        epilog='Example: %s cmac -k key.bin -i data.bin -o cmac.bin'%basename)
    parser_cmac.add_argument('-k', '--key', type=argparse.FileType('rb'), help='Key file(binary)', required=True)
    parser_cmac.add_argument('-i', '--input', type=argparse.FileType('rb'), help='Input file(binary)', required=True)
    parser_cmac.add_argument('-o,', '--output', type=argparse.FileType('wb'), help='Output file(binary)', required=True)
    parser_cmac.set_defaults(func=doCMAC)

    return parser

def main():
    parser = buildParser()
    arguments = parser.parse_args()
    if hasattr(arguments,'func'):
        arguments.func(arguments)
    else:
        parser = buildParser()
        arguments = parser.parse_args(['-h'])


if __name__ == "__main__":
    #test()
    main()
