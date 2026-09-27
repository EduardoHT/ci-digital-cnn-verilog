"""Dataset sintetico reprodutivel e modelo com arquitetura fixa para o RTL.

Conv2d usa correlacao cruzada: nao inverta os kernels ao exportar.
Entrada: PNG uint8, convertida para float32 / 256 (NAO / 255).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["circulo", "quadrado", "triangulo", "cruz", "estrela"]
SEED = 20260925
IMAGE_SIZE = 28
ARCHITECTURE = "cnn28_c8_c16_fc5_v2"
ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def shape_image(label: int, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    """Desenha UMA figura nova, depois antialias, blur e ruido controlados.

    As distribuicoes de posicao, tamanho, intensidade e ruido sao comuns
    as classes. A figura geometrica nao e cortada pelas bordas.
    """
    if label not in range(5):
        raise ValueError("label deve estar entre 0 e 4")
    ss = 4
    radius = float(rng.uniform(7.0, 10.5))
    angle = float(rng.uniform(0, 2 * math.pi))
    cross_ratio = float(rng.uniform(0.32, 0.44))
    star_ratio = float(rng.uniform(0.42, 0.52))
    margin = radius + 1.8
    cx, cy = rng.uniform(margin, 28 - margin, 2)
    fg, bg = int(rng.integers(180, 256)), int(rng.integers(0, 31))
    blur = float(rng.uniform(0, 0.65))
    noise = float(rng.uniform(0, 7))
    canvas = Image.new("L", (28 * ss, 28 * ss), bg)
    draw = ImageDraw.Draw(canvas)
    if label == 0:
        draw.ellipse(((cx-radius)*ss, (cy-radius)*ss,
                      (cx+radius)*ss, (cy+radius)*ss), fill=fg)
    else:
        if label in (1, 2, 4):
            n = {1: 4, 2: 3, 4: 10}[label]
            vertices = []
            for k in range(n):
                r = star_ratio if label == 4 and k % 2 else 1.0
                a = 2 * math.pi * k / n
                vertices.append((r * math.cos(a), r * math.sin(a)))
        else:
            t = cross_ratio
            # Os 12 vertices de uma cruz, normalizados ao raio externo 1.
            vertices = [(-t,-1),(t,-1),(t,-t),(1,-t),(1,t),(t,t),
                        (t,1),(-t,1),(-t,t),(-1,t),(-1,-t),(-t,-t)]
            norm = math.sqrt(1 + t*t)
            vertices = [(x/norm, y/norm) for x, y in vertices]
        co, si = math.cos(angle), math.sin(angle)
        points = [((cx + radius*(x*co-y*si))*ss,
                   (cy + radius*(x*si+y*co))*ss) for x, y in vertices]
        draw.polygon(points, fill=fg)
    canvas = canvas.resize((28, 28), Image.Resampling.LANCZOS)
    canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    pixels = np.asarray(canvas, dtype=np.float64)
    pixels = np.clip(np.rint(pixels + rng.normal(0, noise, (28,28))), 0, 255)
    return Image.fromarray(pixels.astype(np.uint8)), {
        "radius": radius, "angle_deg": math.degrees(angle), "cx": float(cx),
        "cy": float(cy), "foreground": fg, "background": bg,
        "blur_sigma": blur, "noise_std": noise,
        "cross_ratio": cross_ratio, "star_ratio": star_ratio,
    }


def generate_dataset(out: Path = ROOT / "dataset", total: int = 2000,
                     seed: int = SEED, overwrite: bool = False) -> dict:
    """70/15/15, balanceado. Gera cada imagem de forma independente.

    Use outra pasta para um novo dataset. --overwrite so remove os PNGs
    das pastas de classes deste dataset, nunca outros arquivos do projeto.
    """
    out = Path(out)
    if total < 100 or total % 100:
        raise ValueError("Use um total multiplo de 100, por exemplo 1000 ou 2000")
    if (out / "manifest.csv").exists() and not overwrite:
        raise FileExistsError(f"Dataset ja existe em {out}; use outra pasta ou --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    per_class = total // 5
    counts = {"train": per_class*70//100, "val": per_class*15//100,
              "test": per_class*15//100}
    rows, hashes = [], set()
    for si, (split, count) in enumerate(counts.items()):
        for label, name in enumerate(CLASSES):
            folder = out / split / name
            folder.mkdir(parents=True, exist_ok=True)
            if overwrite:
                for old in folder.glob("*.png"):
                    old.unlink()
            for i in range(count):
                # Seeds independentes por split/classe/amostra. Nao ha uma
                # imagem-base compartilhada entre treino e teste.
                local_seed = seed + si*1_000_000 + label*10_000 + i
                image, meta = shape_image(label, np.random.default_rng(local_seed))
                pixel_hash = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                if pixel_hash in hashes:
                    raise RuntimeError("Imagem duplicada detectada")
                hashes.add(pixel_hash)
                file = folder / f"{name}_{i:04d}.png"
                image.save(file)
                rows.append({"file": file.relative_to(out).as_posix(), "split": split,
                             "label": label, "class_name": name, "seed": local_seed,
                             "sha256_pixels": pixel_hash, **meta})
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    info = {"total_images": total, "image_size": [28,28], "mode": "L", "seed": seed,
            "classes": CLASSES, "class_to_idx": dict(zip(CLASSES, range(5))),
            "counts_per_class": counts,
            "counts_total": {k: v*5 for k,v in counts.items()},
            "unique_pixel_hashes": len(hashes),
            "input_divisor": 256.0,
            "scope": "Figuras claras preenchidas em fundo escuro simples. Nao sao imagens reais de drones."}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    make_contact_sheet(out)
    return info


def make_contact_sheet(dataset: Path, output: Path | None = None) -> Path:
    """Mosaico de inspecao, fora das pastas usadas no treinamento."""
    from PIL import ImageFont
    width, height = 760, 650
    board = Image.new("RGB", (width,height), "white")
    d = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=20)
    d.text((24,16), "CI Digital | exemplos do dataset sintetico", fill="black", font=font)
    small = ImageFont.load_default(size=16)
    for row, name in enumerate(CLASSES):
        d.text((24,64+row*113), name, fill="black", font=small)
        files = sorted((dataset / "train" / name).glob("*.png"))[:6]
        for col, file in enumerate(files):
            im = Image.open(file).convert("RGB").resize((84,84), Image.Resampling.NEAREST)
            board.paste(im, (132+col*101, 56+row*113))
    output = output or dataset / "preview.png"
    board.save(output)
    return output


def load_split(dataset: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carrega em ordem de classe explicita, nao pela ordem alfabetica do SO."""
    pixels, labels, paths = [], [], []
    for label, name in enumerate(CLASSES):
        files = sorted((Path(dataset) / split / name).glob("*.png"))
        if not files:
            raise FileNotFoundError(f"Nenhuma imagem em {dataset}/{split}/{name}")
        for path in files:
            with Image.open(path) as im:
                if im.mode != "L" or im.size != (28,28):
                    raise ValueError(f"Formato invalido: {path}; esperado L, 28x28")
                pixels.append(np.asarray(im).copy())
            labels.append(label)
            paths.append(path.relative_to(dataset).as_posix())
    return np.stack(pixels)[:,None], np.asarray(labels, dtype=np.int64), paths


class ShapeCNN(nn.Module):
    """3253 parametros. Dimensoes e canais devem coincidir com tiny_cnn.sv."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,3, padding=0, bias=True)
        self.conv2 = nn.Conv2d(8,16,3, padding=0, bias=True)
        self.pool = nn.MaxPool2d(2,2,ceil_mode=False)
        self.fc = nn.Linear(16*5*5,5, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x))) # 28 -> 26 -> 13
        x = self.pool(torch.relu(self.conv2(x))) # 13 -> 11 -> 5
        return self.fc(x.flatten(1))            # ordem C,H,W; sem softmax


def get_loader(pixels: np.ndarray, labels: np.ndarray, batch: int = 64,
               shuffle: bool = False, seed: int = SEED) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(pixels.astype(np.float32)/256.0), torch.from_numpy(labels))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = np.zeros((5,5), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = cm.diagonal()
    precision = np.divide(tp, cm.sum(0), out=np.zeros(5), where=cm.sum(0)>0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(5), where=cm.sum(1)>0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros(5), where=(precision+recall)>0)
    return {"samples": int(len(y)), "correct": int((y==pred).sum()),
            "accuracy": float((y==pred).mean()), "macro_f1": float(f1.mean()),
            "confusion_matrix": cm.tolist(),
            "per_class": [{"class":name, "precision":float(precision[k]),
                           "recall":float(recall[k]), "f1":float(f1[k]),
                           "support":int(cm[k].sum())} for k,name in enumerate(CLASSES)]}


@torch.no_grad()
def predict_float(model: nn.Module, pixels: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(pixels), 128):
        x = torch.from_numpy(pixels[start:start+128].astype(np.float32)/256.0).to(device)
        outputs.append(model(x).detach().cpu().numpy())
    return np.concatenate(outputs)


def train_model(dataset: Path = ROOT/"dataset", out: Path = ROOT/"runs"/"meu_treino",
                epochs: int = 40, batch: int = 64, lr: float = 0.002,
                seed: int = SEED, device: str = "auto", patience: int = 10,
                overwrite: bool = False) -> dict:
    """Treina, escolhe checkpoint pela validacao, so entao avalia teste.

    O teste nao e usado para early stopping nem para escolher a arquitetura.
    Runs existentes exigem overwrite explicito para evitar perder resultados.
    """
    if epochs < 1 or batch < 1 or patience < 1 or lr <= 0:
        raise ValueError("epochs, batch, patience e lr devem ser positivos")
    seed_everything(seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    out, dataset = Path(out), Path(dataset)
    if (out/"best.pt").exists() and not overwrite:
        raise FileExistsError(f"Treino ja existe: {out}. Use outro --out ou --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    xtr, ytr, _ = load_split(dataset, "train")
    xv, yv, _ = load_split(dataset, "val")
    model = ShapeCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = get_loader(xtr, ytr, batch, True, seed)
    val_loader = get_loader(xv, yv, 128)
    history, best, stalled = [], (float("-inf"), float("inf")), 0
    best_epoch = 0
    for epoch in range(1,epochs+1):
        model.train()
        loss_sum, correct, count = 0., 0, 0
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits,y)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.item())*len(y)
            correct += int((logits.argmax(1)==y).sum().item()); count += len(y)
        model.eval()
        vloss, vc = 0., 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device),y.to(device)
                logits = model(x)
                vloss += float(loss_fn(logits,y).item())*len(y)
                vc += int((logits.argmax(1)==y).sum().item())
        va, vl = vc/len(yv), vloss/len(yv)
        record = {"epoch":epoch,"train_loss":loss_sum/count,"train_accuracy":correct/count,
                  "val_loss":vl,"val_accuracy":va}
        history.append(record)
        print(f"epoch {epoch:02d} loss={record['train_loss']:.4f} "
              f"train={correct/count:.4f} val={va:.4f} val_loss={vl:.4f}", flush=True)
        if va > best[0] or (va == best[0] and vl < best[1]):
            best, stalled, best_epoch = (va,vl), 0, epoch
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, out/"best.pt")
        else:
            stalled += 1
        if stalled >= patience:
            break
    model.load_state_dict(torch.load(out/"best.pt", map_location=device, weights_only=True))
    xt,yt,paths = load_split(dataset,"test")
    test_logits = predict_float(model,xt,device)
    metrics = classification_metrics(yt,test_logits.argmax(1))
    report = {"architecture":ARCHITECTURE,"parameters":sum(p.numel() for p in model.parameters()),
              "seed":seed,"device":device,"epochs_requested":epochs,"epochs_run":len(history),
              "best_epoch":best_epoch,"best_val_accuracy":best[0],"best_val_loss":best[1],
              "float_test":metrics,"input_divisor":256.,"classes":CLASSES,
              "environment":{"python":platform.python_version(), "torch":str(torch.__version__),
                             "numpy":np.__version__},
              "dataset_manifest_sha256":hashlib.sha256((dataset/"manifest.csv").read_bytes()).hexdigest()}
    (out/"training_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False), encoding="utf-8")
    with (out/"history.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    with (out/"test_predictions_float.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["file","true","pred","logit0","logit1","logit2","logit3","logit4"])
        for path,y,logits in zip(paths,yt,test_logits):
            writer.writerow([path,int(y),int(logits.argmax()),*logits.tolist()])
    np.save(out/"float_test_logits.npy",test_logits)
    print(json.dumps(metrics,indent=2,ensure_ascii=False), flush=True)
    return report
