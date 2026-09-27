"""Testes automatizados do contrato numerico e dos arquivos do dataset.

Executar na raiz: python -m unittest discover -s tests -v
Estes testes Python NAO substituem uma simulacao SystemVerilog.
"""
import csv,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from python.shapes import ROOT,CLASSES,ShapeCNN,load_split
from python.quantization import (conv_int,pool_int,requant_relu,write_mem,read_mem,
                                 integer_forward,quantize_model,assert_int32)

class TestNumericContract(unittest.TestCase):
    def test_relu_round_saturate(self):
        x=np.asarray([-2**31,-8,-1,0,3,4,7,8,2044,2**31-1],dtype=np.int64)
        y=requant_relu(x,3)
        np.testing.assert_array_equal(y,[0,0,0,0,0,1,1,1,255,255])
        np.testing.assert_array_equal(requant_relu(np.array([7,255,256]),0),[7,255,255])
    def test_convolution_with_scalar_loop(self):
        rng=np.random.default_rng(4)
        x=rng.integers(0,256,(2,2,5,5),dtype=np.uint8)
        w=rng.integers(-128,128,(3,2,3,3),dtype=np.int16).astype(np.int8)
        b=np.array([-1024,0,403],dtype=np.int32)
        got=conv_int(x,w,b)
        want=np.zeros((2,3,3,3),dtype=np.int64)
        for n in range(2):
            for o in range(3):
                for yy in range(3):
                    for xx in range(3):
                        v=int(b[o])
                        for c in range(2):
                            for ky in range(3):
                                for kx in range(3):
                                    v+=int(x[n,c,yy+ky,xx+kx])*int(w[o,c,ky,kx])
                        want[n,o,yy,xx]=v
        np.testing.assert_array_equal(got,want)
        # Conv2d float64 tambem e exata neste pequeno intervalo de inteiros.
        ref=torch.nn.functional.conv2d(torch.tensor(x,dtype=torch.float64),
                torch.tensor(w,dtype=torch.float64),torch.tensor(b,dtype=torch.float64))
        np.testing.assert_array_equal(got,ref.numpy())
    def test_pooling_odd_border(self):
        x=np.zeros((1,1,11,11),dtype=np.uint8);x[:,:,10,:]=255;x[:,:,:,10]=255
        np.testing.assert_array_equal(pool_int(x),np.zeros((1,1,5,5)))
        x[0,0,1,1]=230
        self.assertEqual(int(pool_int(x)[0,0,0,0]),230)
    def test_mem_signed_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bits,vals in [(8,[-128,-1,0,127]),(32,[-2**31,-1,0,2**31-1])]:
                path=Path(tmp)/'values.mem';write_mem(path,np.array(vals),bits)
                np.testing.assert_array_equal(read_mem(path,bits,True),vals)
    def test_architecture(self):
        model=ShapeCNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()),3253)
        self.assertEqual(tuple(model(torch.zeros(2,1,28,28)).shape),(2,5))
    def test_int32_overflow_detected(self):
        with self.assertRaises(OverflowError):assert_int32(np.array([2**31]),'test')
    def test_model_batch_equivalence(self):
        q=dict(np.load(ROOT/'exports'/'trained'/'quantized_model.npz'))
        x,_,_=load_split(ROOT/'dataset','test')
        batched=integer_forward(x[[0,60,120]],q)
        for j,i in enumerate([0,60,120]):
            np.testing.assert_array_equal(batched[j],integer_forward(x[i:i+1],q)[0])
    def test_exported_expected_first_sample(self):
        root=ROOT/'exports'/'trained';q=dict(np.load(root/'quantized_model.npz'))
        x=read_mem(root/'vectors'/'images.mem',8)[:784].astype(np.uint8).reshape(1,1,28,28)
        tr=integer_forward(x,q,True)
        for name,a in tr.items():
            n=a.size;bits=32 if name=='logits' else 8
            exp=read_mem(root/'vectors'/f'expected_{name}.mem',bits,name=='logits')[:n]
            np.testing.assert_array_equal(a.ravel(),exp)

class TestDataset(unittest.TestCase):
    def test_manifest_counts_hashes_and_format(self):
        root=ROOT/'dataset'
        with (root/'manifest.csv').open(encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),2000)
        self.assertEqual(len(set(r['sha256_pixels'] for r in rows)),2000)
        self.assertEqual(len(list(root.glob('*/*/*.png'))),2000)
        expected={'train':280,'val':60,'test':60}
        for split,n in expected.items():
            for name in CLASSES:
                self.assertEqual(sum(r['split']==split and r['class_name']==name for r in rows),n)
        for r in rows:
            with Image.open(root/r['file']) as im:
                self.assertEqual(im.mode,'L');self.assertEqual(im.size,(28,28))
                self.assertEqual(hashlib.sha256(np.asarray(im).tobytes()).hexdigest(),r['sha256_pixels'])

if __name__=='__main__':unittest.main()
