"""Gera design.sv + testbench.sv autocontidos para o EDA Playground.

Os mesmos modulos RTL sao concatenados; o testbench embute dados em constantes
hexadecimais. Nada de copiar manualmente milhares de pesos ou enviar 2000 PNGs.
"""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from python.quantization import read_mem


def packed(name: str, arr: np.ndarray, bits: int) -> str:
    # Elemento 0 nos bits menos significativos, para leitura [j*bits +: bits].
    a=np.asarray(arr).reshape(-1)
    mask=(1<<bits)-1
    text=''.join(f'{int(x)&mask:0{bits//4}x}' for x in a[::-1])
    parts=[text[i:i+128] for i in range(0,len(text),128)]
    return (f'    localparam [{len(a)*bits-1}:0] {name} = {{\n        '+
            ',\n        '.join(f"{len(part)*4}'h{part}" for part in parts)+'\n    };\n')


def make_eda(export: Path, out: Path, samples: int = 5) -> dict:
    export=Path(export).resolve();out=Path(out).resolve()
    available=json.loads((export/'vectors'/'vectors.json').read_text())['n_vectors']
    if not 1<=samples<=available:raise ValueError(f'Use 1..{available} amostras')
    if samples>20:raise ValueError('Limite didatico de 20 amostras por pacote online')
    source=(ROOT/'tb'/'tb_cnn.sv').read_text()
    source=source.replace('parameter integer NTEST=20',f'parameter integer NTEST={samples}')
    source=source.replace('.LOAD_FILES(1)', '.LOAD_FILES(0)')
    specs=[]
    for name,n,bits in [('conv1_w',72,8),('conv1_b',8,32),('conv2_w',1152,8),
                        ('conv2_b',16,32),('fc_w',2000,8),('fc_b',5,32),('shifts',2,8)]:
        specs.append((name.upper(),read_mem(export/f'{name}.mem',bits),bits,f'dut.{name}'))
    vectors=[('images',784,8,'images'),('labels',1,8,'labels'),
             ('expected_conv1',5408,8,'exp_c1'),('expected_pool1',1352,8,'exp_p1'),
             ('expected_conv2',1936,8,'exp_c2'),('expected_pool2',400,8,'exp_p2'),
             ('expected_logits',5,32,'exp_logits'),('expected_class',1,8,'exp_class')]
    for name,n,bits,dest in vectors:
        specs.append((name.upper(),read_mem(export/'vectors'/f'{name}.mem',bits)[:samples*n],bits,dest))
    constants='\n'.join(packed(name,arr,bits) for name,arr,bits,_ in specs)
    initialization='    initial begin\n'
    for name,arr,bits,dest in specs:
        initialization += f'        for(j=0;j<{len(arr)};j=j+1) {dest}[j]={name}[j*{bits} +: {bits}];\n'
    st=source.index('    initial begin\n        require_file')
    en=source.index('        if ($test$plusargs("WAVES"))',st)
    source=source[:st]+initialization+source[en:]
    source=source.replace('    task require_file', constants+'\n    task require_file')
    source='// AUTOGERADO por scripts/make_eda.py. Exportacao: '+export.name+'\n'+source
    modules=['mac_u8s8.sv','relu_requant.sv','max4_u8.sv','tiny_cnn.sv']
    design='\n\n'.join((ROOT/'rtl'/name).read_text() for name in modules)
    out.mkdir(parents=True,exist_ok=True)
    (out/'design.sv').write_text(design,encoding='utf-8')
    (out/'testbench.sv').write_text(source,encoding='utf-8')
    chars=len(source)+len(design)
    if chars>1_000_000:raise ValueError('Codigo excede limite documentado do EDA Playground')
    info={'export_name':export.name,'samples':samples,'total_characters':chars,
          'full_intermediate_comparison':True,'external_mem_files_required':False,
          'execution_status':'NOT_EXECUTED_BY_THIS_GENERATOR'}
    (out/'package_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))
    return info

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=5)
    a=p.parse_args();make_eda(a.export,a.out,a.samples)
