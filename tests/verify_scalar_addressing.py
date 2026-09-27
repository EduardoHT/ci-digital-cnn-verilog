"""Verificacao independente do enderecamento usado no RTL, em Python escalar.

NAO e um simulador SystemVerilog. Nao verifica sintaxe nem semantica HDL.
Compara enderecos linearizados e todas as operacoes com os .mem de referencia.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from python.quantization import read_mem


def scalar_forward(image: np.ndarray, q: dict) -> dict:
    img=list(map(int,image.ravel()))
    cw1=list(map(int,q['conv1_w'].ravel()));cb1=list(map(int,q['conv1_b']))
    cw2=list(map(int,q['conv2_w'].ravel()));cb2=list(map(int,q['conv2_b']))
    fw=list(map(int,q['fc_w'].ravel()));fb=list(map(int,q['fc_b']))
    shifts=list(map(int,q['shifts']))
    c1=[0]*5408;p1=[0]*1352;c2=[0]*1936;p2=[0]*400;logits=[0]*5
    def rq(acc,shift):
        if not -(2**31)<=acc<2**31:raise OverflowError('scalar int32')
        if acc<=0:return 0
        return min(255,((acc+(1<<(shift-1)))>>shift) if shift else acc)
    for out in range(5408):
        oc=out//676;oy=(out%676)//26;ox=out%26;acc=cb1[oc]
        for tap in range(9):
            addr=(oy+tap//3)*28+ox+tap%3
            acc+=img[addr]*cw1[oc*9+tap]
        c1[out]=rq(acc,shifts[0])
    for idx in range(1352):
        c=idx//169;py=(idx%169)//13;px=idx%13
        base=c*676+2*py*26+2*px
        p1[idx]=max(c1[base],c1[base+1],c1[base+26],c1[base+27])
    for out in range(1936):
        oc=out//121;oy=(out%121)//11;ox=out%11;acc=cb2[oc]
        for tap in range(72):
            ic=tap//9;ky=(tap%9)//3;kx=tap%3
            addr=ic*169+(oy+ky)*13+ox+kx
            acc+=p1[addr]*cw2[oc*72+tap]
        c2[out]=rq(acc,shifts[1])
    for idx in range(400):
        c=idx//25;py=(idx%25)//5;px=idx%5
        base=c*121+2*py*11+2*px
        p2[idx]=max(c2[base],c2[base+1],c2[base+11],c2[base+12])
    for out in range(5):
        acc=fb[out]
        for tap in range(400):acc+=p2[tap]*fw[out*400+tap]
        if not -(2**31)<=acc<2**31:raise OverflowError('dense int32')
        logits[out]=acc
    best=0
    for i in range(1,5):
        if logits[i]>logits[best]:best=i
    return {'conv1':c1,'pool1':p1,'conv2':c2,'pool2':p2,'logits':logits,'class':best}


def verify(folder: Path) -> dict:
    q=dict(np.load(folder/'quantized_model.npz'));v=folder/'vectors'
    n=json.loads((v/'vectors.json').read_text())['n_vectors']
    images=read_mem(v/'images.mem',8).reshape(n,784)
    expected={}
    for name,size,bits in [('conv1',5408,8),('pool1',1352,8),('conv2',1936,8),('pool2',400,8),('logits',5,32),('class',1,8)]:
        expected[name]=read_mem(v/f'expected_{name}.mem',bits,name=='logits').reshape(n,size)
    for t,image in enumerate(images):
        actual=scalar_forward(image,q)
        for key in actual:np.testing.assert_array_equal(np.asarray(actual[key]).ravel(),expected[key][t],err_msg=f'{folder.name} t={t} {key}')
        print(f'PASS_SCALAR mode={folder.name} vector={t}',flush=True)
    return {'mode':folder.name,'vectors':n,'status':'PASS','values_compared':n*9102,
            'tool':'Python scalar addressing model, NOT HDL simulator'}

if __name__=='__main__':
    reports=[verify(ROOT/'exports'/name) for name in ['trained','demo']]
    (ROOT/'reports'/'scalar_address_verification.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))
