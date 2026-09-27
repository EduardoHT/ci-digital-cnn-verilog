"""Referencia inteira explicita e exportacao para o SystemVerilog.

Conv1/Conv2: uint8 x int8 -> acumulador int32 -> ReLU -> arredonda -> satura uint8.
Dense: uint8 x int8 -> int32; logits das 5 classes na MESMA escala.
Python usa int64 internamente para detectar, nao esconder, estouro de int32.
"""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from .shapes import (ROOT, SEED, ARCHITECTURE, CLASSES, ShapeCNN, load_split,
                     classification_metrics, predict_float, seed_everything)

SPECS = [("conv1", (8,1,3,3)), ("conv2", (16,8,3,3)), ("fc", (5,400))]
I32_MIN, I32_MAX = -(2**31), 2**31 - 1


def assert_int32(a: np.ndarray, name: str) -> None:
    if np.min(a) < I32_MIN or np.max(a) > I32_MAX:
        raise OverflowError(f"{name} excede int32")


def choose_fraction(max_abs: float, limit: int, low: int = -8, high: int = 16) -> int:
    if not math.isfinite(max_abs):
        raise ValueError("Tensor contem NaN ou infinito")
    if max_abs < 1e-12:
        return 8
    return int(np.clip(math.floor(math.log2(limit / max_abs)), low, high))


def requant_relu(acc: np.ndarray, shift: int) -> np.ndarray:
    """Meio para cima APOS ReLU, com saturacao e intermediario int64.

    Ex.: shift=3, acc=4 -> 1; acc=3 -> 0; acc=-4 -> 0.
    Nao usa round() para as ativacoes, pois round tem outra regra de empate.
    """
    if not 0 <= shift <= 30:
        raise ValueError("shift deve estar entre 0 e 30")
    x = np.maximum(np.asarray(acc,dtype=np.int64), 0)
    if shift:
        x = (x + (1 << (shift-1))) >> shift
    return np.clip(x,0,255).astype(np.uint8)


def conv_int(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlacao cruzada valid, stride=1. NCHW; peso OIHW."""
    x = np.asarray(x,dtype=np.int64)
    w = np.asarray(w,dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(x,(3,3),axis=(-2,-1))
    acc = np.einsum('nchwkl,ockl->nohw',windows,w,optimize=True,dtype=np.int64)
    acc += np.asarray(b,dtype=np.int64)[None,:,None,None]
    assert_int32(acc, "conv")
    return acc


def pool_int(x: np.ndarray) -> np.ndarray:
    """MaxPool2x2, stride=2, ceil_mode=False. Ignora sobra impar."""
    n,c,h,w=x.shape
    h2,w2=h//2,w//2
    return x[:,:,:h2*2,:w2*2].reshape(n,c,h2,2,w2,2).max(axis=(3,5))


def integer_forward(pixels: np.ndarray, q: dict, trace: bool = False):
    x=np.asarray(pixels)
    if x.ndim == 2: x=x[None,None]
    if x.ndim != 4 or x.shape[1:] != (1,28,28) or x.dtype != np.uint8:
        raise ValueError("Esperado uint8 N x 1 x 28 x 28, sem normalizacao float")
    c1=requant_relu(conv_int(x,q['conv1_w'],q['conv1_b']),int(q['shifts'][0]))
    p1=pool_int(c1)
    c2=requant_relu(conv_int(p1,q['conv2_w'],q['conv2_b']),int(q['shifts'][1]))
    p2=pool_int(c2)
    logits=p2.reshape(len(x),400).astype(np.int64) @ q['fc_w'].astype(np.int64).T
    logits+=q['fc_b'].astype(np.int64)
    assert_int32(logits,"dense")
    if trace:
        return {"conv1":c1,"pool1":p1,"conv2":c2,"pool2":p2,"logits":logits.astype(np.int32)}
    return logits.astype(np.int32)


def quantize_model(model: ShapeCNN, calibration_pixels: np.ndarray) -> tuple[dict,dict]:
    """Calibra apenas com treino. Escalas sao potencias de dois por tensor.

    Pesos/bias: np.rint (empate para par) somente na exportacao offline.
    Ativacoes no RTL: ReLU, half-up, shift, saturacao; ver requant_relu().
    """
    model=model.cpu().eval()
    maxima=[0.,0.]
    with torch.no_grad():
        for st in range(0,len(calibration_pixels),64):
            x=torch.from_numpy(calibration_pixels[st:st+64].astype(np.float32)/256.)
            a=torch.relu(model.conv1(x)); maxima[0]=max(maxima[0],float(a.max()))
            a=torch.relu(model.conv2(model.pool(a))); maxima[1]=max(maxima[1],float(a.max()))
    # 240 em vez de 255 deixa pequena margem, sem olhar o conjunto de teste.
    af=[choose_fraction(m,240) for m in maxima]
    q, layers = {}, []
    shifts=[]
    for i,(name,shape) in enumerate(SPECS):
        layer=getattr(model,name)
        weights=layer.weight.detach().numpy().astype(np.float64)
        bias=layer.bias.detach().numpy().astype(np.float64)
        if weights.shape != shape: raise ValueError(f"Arquitetura divergente em {name}")
        wf=choose_fraction(float(np.max(np.abs(weights))),127)
        inf=8 if i==0 else af[i-1]
        qw=np.clip(np.rint(weights*(2.**wf)),-127,127).astype(np.int8)
        qb=np.rint(bias*(2.**(inf+wf))).astype(np.int64)
        assert_int32(qb,name+" bias")
        # Limite por saida, valido para QUALQUER vetor uint8 de entrada.
        fan_in=int(np.prod(shape[1:]))
        bounds=np.abs(qb)+255*np.abs(qw.astype(np.int64).reshape(shape[0],-1)).sum(1)
        if int(bounds.max()) > I32_MAX:
            raise OverflowError(f"Limite de acumulacao inseguro em {name}")
        q[name+'_w'],q[name+'_b']=qw,qb.astype(np.int32)
        record={"name":name,"shape":list(shape),"weight_fraction_bits":wf,
                "input_fraction_bits":inf,"bias_fraction_bits":inf+wf,
                "worst_case_abs_accumulator_bound":int(bounds.max()),"fan_in":fan_in}
        if i<2:
            shift=inf+wf-af[i]
            if not 0<=shift<=30:
                raise ValueError(f"Shift {shift} nao suportado; revise escalas")
            shifts.append(shift)
            record.update(output_fraction_bits=af[i],right_shift=shift,
                          calibration_float_max=maxima[i])
        else:
            record['logit_scale']=2.**(-(inf+wf))
        layers.append(record)
    q['shifts']=np.asarray(shifts,dtype=np.uint8)
    return q,{"architecture":ARCHITECTURE,"input_divisor":256.,"classes":CLASSES,
              "round_weights_bias":"nearest ties to even (np.rint), offline",
              "round_activations":"ReLU -> half up -> right shift -> saturate [0,255]",
              "weight_dtype":"int8 [-127,127]","activation_dtype":"uint8",
              "bias_accumulator_dtype":"int32","flatten":"C,H,W, row-major",
              "convolution":"cross-correlation valid (kernel is NOT flipped)",
              "calibration_samples":len(calibration_pixels),"calibration_split":"train",
              "layers":layers}


def write_mem(path: Path, values: np.ndarray, bits: int) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    a=np.asarray(values).reshape(-1)
    mask=(1<<bits)-1
    path.write_text(''.join(f'{int(x)&mask:0{bits//4}x}\n' for x in a),encoding='ascii')


def read_mem(path: Path, bits: int, signed: bool = False) -> np.ndarray:
    vals=np.asarray([int(v,16) for v in Path(path).read_text().split()],dtype=np.int64)
    if signed: vals=np.where(vals>=(1<<(bits-1)),vals-(1<<bits),vals)
    return vals


def export_parameters(out: Path, q: dict, meta: dict) -> None:
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,_ in SPECS:
        write_mem(out/f'{name}_w.mem',q[name+'_w'],8)
        write_mem(out/f'{name}_b.mem',q[name+'_b'],32)
    write_mem(out/'shifts.mem',q['shifts'],8)
    np.savez_compressed(out/'quantized_model.npz',**q)
    (out/'quantization.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    # Verifica arquivos reais, nao apenas os tensores antes de salvar.
    for name,shape in SPECS:
        rw=read_mem(out/f'{name}_w.mem',8,True).reshape(shape)
        rb=read_mem(out/f'{name}_b.mem',32,True)
        np.testing.assert_array_equal(rw,q[name+'_w'])
        np.testing.assert_array_equal(rb,q[name+'_b'])
    np.testing.assert_array_equal(read_mem(out/'shifts.mem',8),q['shifts'])


def boundary_images() -> tuple[np.ndarray,list[str]]:
    """Estresses aritmeticos, sem rotulo semantico; NAO contam na acuracia."""
    a=np.zeros((5,1,28,28),dtype=np.uint8)
    a[1]=255
    a[2,0]=(np.indices((28,28)).sum(0)%2)*255
    a[3,0,14,14]=255
    a[4]=np.random.default_rng(991).integers(0,256,(1,28,28),dtype=np.uint8)
    return a,['zeros','all_255','checkerboard','impulse','uniform_noise']


def export_vectors(out: Path, q: dict, pixels: np.ndarray, labels: np.ndarray,
                   paths: list[str], per_class: int = 3, add_edges: bool = True) -> dict:
    if per_class<1: raise ValueError("per_class deve ser >= 1")
    # Intercala classes para que primeiros 5 vetores cubram todas as classes.
    selected=[]
    for j in range(per_class):
        for c in range(5):
            idx=np.flatnonzero(labels==c)
            if len(idx)<=j: raise ValueError("Poucas imagens na classe")
            selected.append(int(idx[j]))
    x=pixels[selected]; y=labels[selected]
    samples=[{"file":paths[i],"true":int(labels[i]),"kind":"test"} for i in selected]
    if add_edges:
        edge,names=boundary_images(); x=np.concatenate([x,edge])
        y=np.concatenate([y,np.full(len(edge),255,dtype=np.int64)])
        samples += [{"file":name,"true":None,"kind":"arithmetic_edge"} for name in names]
    tr=integer_forward(x,q,trace=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_mem(out/'images.mem',x,8);write_mem(out/'labels.mem',y,8)
    for name,arr in tr.items():write_mem(out/f'expected_{name}.mem',arr,32 if name=='logits' else 8)
    write_mem(out/'expected_class.mem',tr['logits'].argmax(1),8)
    for s,logits in zip(samples,tr['logits']):
        s['expected_class']=int(logits.argmax());s['expected_logits']=logits.tolist()
    info={"n_vectors":len(x),"n_labeled":len(selected),"n_edges":len(x)-len(selected),
          "per_sample_values":{"images":784,"conv1":5408,"pool1":1352,"conv2":1936,"pool2":400,"logits":5},
          "samples":samples}
    (out/'vectors.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return info


def run_export(checkpoint: Path | None, out: Path, dataset: Path = ROOT/'dataset',
               per_class: int = 3, random_demo: bool = False, overwrite: bool = False) -> dict:
    out=Path(out);dataset=Path(dataset)
    if (out/'quantization.json').exists() and not overwrite:
        raise FileExistsError(f"Exportacao existe: {out}; use outra pasta ou --overwrite")
    seed_everything(SEED)
    model=ShapeCNN()
    if not random_demo:
        if checkpoint is None: raise ValueError("Forneca checkpoint ou --random-demo")
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    train_x,_,_=load_split(dataset,'train')
    calibration_idx=np.random.default_rng(SEED+33).permutation(len(train_x))[:500]
    q,meta=quantize_model(model,train_x[calibration_idx])
    meta['mode']='random_demo_not_trained' if random_demo else 'trained'
    meta['checkpoint_sha256']=None if random_demo else hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    meta['calibration_indices']=calibration_idx.tolist()
    export_parameters(out,q,meta)
    pixels,y,paths=load_split(dataset,'test')
    float_logits=predict_float(model,pixels)
    integer_logits=[];saturation_counts={"conv1":0,"conv2":0};activation_counts={"conv1":0,"conv2":0}
    for start in range(0,len(pixels),32):
        trace=integer_forward(pixels[start:start+32],q,True)
        integer_logits.append(trace['logits'])
        for name in saturation_counts:
            # Valor no teto: inclui eventual 255 obtido sem clipping.
            saturation_counts[name]+=int((trace[name]==255).sum())
            activation_counts[name]+=int(trace[name].size)
    integer_logits=np.concatenate(integer_logits)
    report={"mode":meta['mode'],"float_test":classification_metrics(y,float_logits.argmax(1)),
            "integer_test":classification_metrics(y,integer_logits.argmax(1)),
            "float_int_class_agreement":float((float_logits.argmax(1)==integer_logits.argmax(1)).mean()),
            "activation_at_255_fraction":{k:saturation_counts[k]/activation_counts[k] for k in saturation_counts},
            "note":"Equivalencia RTL deve ser com esta referencia inteira, nao com float32.",
            "mem_roundtrip":"PASS"}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    np.save(out/'integer_test_logits.npy',integer_logits)
    vectors=export_vectors(out/'vectors',q,pixels,y,paths,per_class,add_edges=True)
    (out/'export_summary.json').write_text(json.dumps({'n_vectors':vectors['n_vectors'],'mode':meta['mode']},indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['float_test','integer_test']},indent=2),flush=True)
    print('float accuracy',report['float_test']['accuracy'],'integer accuracy',report['integer_test']['accuracy'],flush=True)
    return report
