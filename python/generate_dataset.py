"""Execute na raiz: python -m python.generate_dataset --total 2000"""
from pathlib import Path
import argparse, json
from .shapes import ROOT, SEED, generate_dataset

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'dataset')
    p.add_argument('--total',type=int,default=2000)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    print(json.dumps(generate_dataset(a.out,a.total,a.seed,a.overwrite),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
