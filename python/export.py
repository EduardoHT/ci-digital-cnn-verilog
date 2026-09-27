"""Quantiza, salva pesos/bias/shifts .mem e vetores completos de verificacao."""
from pathlib import Path
import argparse
from .shapes import ROOT
from .quantization import run_export

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=ROOT/'dataset')
    p.add_argument('--per-class',type=int,default=3)
    p.add_argument('--random-demo',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();run_export(**vars(a))
if __name__=='__main__': main()
