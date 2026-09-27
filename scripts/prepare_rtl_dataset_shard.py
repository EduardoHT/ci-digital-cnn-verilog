"""Prepara um lote do dataset para inferencia SystemVerilog com pesos demo."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["circulo", "quadrado", "triangulo", "cruz", "estrela"]
I32_MIN, I32_MAX = -(2**31), 2**31 - 1


def requant_relu(acc: np.ndarray, shift: int) -> np.ndarray:
    x = np.maximum(np.asarray(acc, dtype=np.int64), 0)
    if shift:
        x = (x + (1 << (shift - 1))) >> shift
    return np.clip(x, 0, 255).astype(np.uint8)


def conv_int(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.int64)
    w64 = np.asarray(w, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(x64, (3, 3), axis=(-2, -1))
    acc = np.einsum("nchwkl,ockl->nohw", windows, w64, optimize=True, dtype=np.int64)
    acc += np.asarray(b, dtype=np.int64)[None, :, None, None]
    if int(acc.min()) < I32_MIN or int(acc.max()) > I32_MAX:
        raise OverflowError("Acumulador de convolucao excedeu int32")
    return acc


def pool_int(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    return x[:, :, : h2 * 2, : w2 * 2].reshape(n, c, h2, 2, w2, 2).max(axis=(3, 5))


def integer_forward(pixels: np.ndarray, q: dict[str, np.ndarray]) -> np.ndarray:
    c1 = requant_relu(conv_int(pixels, q["conv1_w"], q["conv1_b"]), int(q["shifts"][0]))
    p1 = pool_int(c1)
    c2 = requant_relu(conv_int(p1, q["conv2_w"], q["conv2_b"]), int(q["shifts"][1]))
    p2 = pool_int(c2)
    logits = p2.reshape(len(pixels), 400).astype(np.int64) @ q["fc_w"].astype(np.int64).T
    logits += q["fc_b"].astype(np.int64)
    if int(logits.min()) < I32_MIN or int(logits.max()) > I32_MAX:
        raise OverflowError("Logits excederam int32")
    return logits.astype(np.int32)


def write_mem(path: Path, values: np.ndarray, bits: int) -> None:
    mask = (1 << bits) - 1
    width = bits // 4
    flat = np.asarray(values).reshape(-1)
    path.write_text("".join(f"{int(v) & mask:0{width}x}\n" for v in flat), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Indice de lote invalido")

    manifest = ROOT / "dataset" / "manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    if len(rows) != 2000:
        raise RuntimeError(f"Esperadas 2000 imagens, encontradas {len(rows)}")
    paths = [r["file"] for r in rows]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Manifesto contem caminhos duplicados")
    for row in rows:
        label = int(row["label"])
        if not 0 <= label < 5 or row["class_name"] != CLASSES[label]:
            raise RuntimeError(f"Rotulo/classe inconsistente: {row}")

    start = len(rows) * args.shard_index // args.shard_count
    end = len(rows) * (args.shard_index + 1) // args.shard_count
    selected = rows[start:end]
    pixels = np.empty((len(selected), 1, 28, 28), dtype=np.uint8)
    for local_index, row in enumerate(selected):
        image_path = ROOT / "dataset" / row["file"]
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        if image.shape != (28, 28):
            raise RuntimeError(f"Dimensao invalida em {row['file']}: {image.shape}")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest != row["sha256_pixels"]:
            raise RuntimeError(f"Hash de pixels divergente em {row['file']}")
        pixels[local_index, 0] = image

    with np.load(ROOT / "exports" / "demo" / "quantized_model.npz") as archive:
        q = {name: archive[name] for name in archive.files}
    logits = integer_forward(pixels, q)
    pred = logits.argmax(axis=1).astype(np.uint8)
    labels = np.asarray([int(r["label"]) for r in selected], dtype=np.uint8)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_mem(out / "images.mem", pixels, 8)
    write_mem(out / "labels.mem", labels, 8)
    write_mem(out / "expected_logits.mem", logits, 32)
    write_mem(out / "expected_class.mem", pred, 8)
    with (out / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["local_index", "global_index", "split", "id", "file", "true", "reference_pred"])
        writer.writeheader()
        for local_index, (row, expected) in enumerate(zip(selected, pred, strict=True)):
            writer.writerow({
                "local_index": local_index,
                "global_index": start + local_index,
                "split": row["split"],
                "id": Path(row["file"]).stem,
                "file": row["file"],
                "true": row["label"],
                "reference_pred": int(expected),
            })
    summary = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "start": start,
        "end": end,
        "n_vectors": len(selected),
        "weights": "exports/demo",
        "reference": "numpy integer bit-compatible",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
