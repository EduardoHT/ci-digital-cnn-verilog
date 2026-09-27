"""Execute na raiz: python -m python.train --out runs/meu_treino"""
from pathlib import Path
import argparse
from .shapes import ROOT,SEED,train_model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'dataset')
    p.add_argument('--out',type=Path,default=ROOT/'runs'/'meu_treino')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();train_model(**vars(a))
if __name__=='__main__': main()
