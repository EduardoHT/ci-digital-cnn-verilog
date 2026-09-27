"""Compila e executa a verificacao REAL com Icarus Verilog instalado no PATH.

Uso: python scripts/run_sim.py --export exports/trained
Nao confundir o PASS esperado com um log de simulacao ja executado.
"""
from __future__ import annotations
import argparse,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RTL=[ROOT/'rtl'/n for n in ['mac_u8s8.sv','relu_requant.sv','max4_u8.sv','tiny_cnn.sv']]

def execute(cmd: list[str], cwd: Path, log: Path, timeout: int = 600):
    print(' '.join(cmd),flush=True)
    r=subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    log.write_text(r.stdout,encoding='utf-8');print(r.stdout,flush=True)
    if r.returncode:raise RuntimeError(f'Falha ({r.returncode}). Veja {log}')
    return r.stdout

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,default=ROOT/'exports'/'trained')
    p.add_argument('--waves',action='store_true')
    a=p.parse_args()
    iv,vvp=shutil.which('iverilog'),shutil.which('vvp')
    if not iv or not vvp:
        raise SystemExit('Icarus Verilog nao encontrado (iverilog e vvp). Use eda_playground/ no navegador ou instale o simulador conforme o PDF. Nenhuma simulacao foi executada.')
    export=a.export.resolve();vectors=export/'vectors'
    n=json.loads((vectors/'vectors.json').read_text())['n_vectors']
    build=ROOT/'build'/export.name;build.mkdir(parents=True,exist_ok=True)
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    compile_arith=[iv,'-g2012','-s','tb_arithmetic','-o',str(build/'arithmetic.vvp'),
                   *map(str,RTL[:3]),str(ROOT/'tb'/'tb_arithmetic.sv')]
    execute(compile_arith,ROOT,reports/f'{export.name}_arithmetic_compile.log')
    alog=execute([vvp,str(build/'arithmetic.vvp')],ROOT,reports/f'{export.name}_arithmetic_simulation.log')
    if 'PASS_ARITHMETIC' not in alog:raise RuntimeError('Teste aritmetico nao concluiu')
    binary=build/'cnn.vvp'
    cmd=[iv,'-g2012','-s','tb_cnn',f'-Ptb_cnn.NTEST={n}',
         f'-Ptb_cnn.WEIGHTS_DIR="{export.as_posix()}"',
         f'-Ptb_cnn.VECTOR_DIR="{vectors.as_posix()}"','-o',str(binary),
         *map(str,RTL),str(ROOT/'tb'/'tb_cnn.sv')]
    execute(cmd,ROOT,reports/f'{export.name}_compile.log')
    log=execute([vvp,str(binary),*(['+WAVES'] if a.waves else [])],ROOT,
                reports/f'{export.name}_simulation.log')
    if 'PASS_ALL' not in log:raise RuntimeError('Simulacao terminou sem PASS_ALL')
    result={'status':'PASS','simulator':'Icarus Verilog','vectors':n,'export':str(export),
            'verified':'feature maps, pooling, logits, argmax, done pulse, cycle count',
            'expected_cycles':206520}
    (reports/f'{export.name}_rtl_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
