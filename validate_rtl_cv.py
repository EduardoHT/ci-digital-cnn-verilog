"""Run the trained baseline and five CV checkpoints through real SystemVerilog."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
RUNS = ["baseline", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
CLASSES = ["circulo", "quadrado", "triangulo", "cruz", "estrela"]
RTL = [ROOT / "rtl" / n for n in ["mac_u8s8.sv", "relu_requant.sv", "max4_u8.sv", "tiny_cnn.sv"]]
OUT_ROOT = ROOT / "results" / "rtl_cv_70_20_10"
SEED = 20260925


def requant_relu(acc, shift):
    x = np.maximum(np.asarray(acc, dtype=np.int64), 0)
    if shift:
        x = (x + (1 << (shift - 1))) >> shift
    return np.clip(x, 0, 255).astype(np.uint8)


def conv_int(x, w, b):
    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(x, dtype=np.int64), (3, 3), axis=(-2, -1))
    acc = np.einsum("nchwkl,ockl->nohw", windows,
                    np.asarray(w, dtype=np.int64), optimize=True, dtype=np.int64)
    return acc + np.asarray(b, dtype=np.int64)[None, :, None, None]


def pool_int(x):
    n, c, h, w = x.shape
    return x[:, :, :h // 2 * 2, :w // 2 * 2].reshape(
        n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def integer_forward(pixels, q):
    c1 = requant_relu(conv_int(pixels, q["conv1_w"], q["conv1_b"]), int(q["shifts"][0]))
    p1 = pool_int(c1)
    c2 = requant_relu(conv_int(p1, q["conv2_w"], q["conv2_b"]), int(q["shifts"][1]))
    p2 = pool_int(c2)
    logits = p2.reshape(len(pixels), 400).astype(np.int64) @ q["fc_w"].astype(np.int64).T
    logits += q["fc_b"].astype(np.int64)
    return logits.astype(np.int32)


def write_mem(path, values, bits):
    mask = (1 << bits) - 1; width = bits // 4
    path.write_text("".join(f"{int(v) & mask:0{width}x}\n" for v in np.asarray(values).reshape(-1)), encoding="ascii")


def metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    total = int(cm.sum()); correct = int(np.trace(cm))
    pc = []
    for i, name in enumerate(CLASSES):
        tp = int(cm[i, i]); support = int(cm[i].sum()); predicted = int(cm[:, i].sum())
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        pc.append({"class": name, "support": support, "predicted": predicted,
                   "tp": tp, "precision": p, "recall": r, "f1": f1})
    return {"total": total, "correct": correct, "accuracy": correct / total,
            "macro_precision": float(np.mean([x["precision"] for x in pc])),
            "macro_recall": float(np.mean([x["recall"] for x in pc])),
            "macro_f1": float(np.mean([x["f1"] for x in pc])), "per_class": pc}


def plot_matrix(cm, path, title):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel("Classe prevista"); ax.set_ylabel("Classe verdadeira"); ax.set_title(title)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def prepare(run):
    run_dir = ROOT / "runs" / "cv_70_20_10" / run
    export_dir = ROOT / "exports" / "cv_70_20_10" / run
    out = OUT_ROOT / run
    vectors = out / "vectors"; vectors.mkdir(parents=True, exist_ok=True)
    all_rows = list(csv.DictReader((ROOT / "dataset" / "manifest.csv").open(encoding="utf-8")))
    all_labels = np.asarray([int(r["label"]) for r in all_rows])
    rng = np.random.default_rng(SEED)
    test_indices = []
    for cls in range(5):
        idx = np.flatnonzero(all_labels == cls)
        rng.shuffle(idx)
        test_indices.extend(idx[:40])
    rows = [all_rows[i] for i in sorted(test_indices)]
    if len(rows) != 200:
        raise RuntimeError(f"{run}: expected 200 fixed-test rows, found {len(rows)}")
    pixels = np.empty((200, 1, 28, 28), dtype=np.uint8)
    labels = np.empty(200, dtype=np.uint8)
    for i, row in enumerate(rows):
        arr = np.asarray(Image.open(ROOT / "dataset" / row["file"]).convert("L"), dtype=np.uint8)
        if arr.shape != (28, 28): raise RuntimeError(f"bad shape {row['file']}: {arr.shape}")
        pixels[i, 0] = arr; labels[i] = int(row["label"])
    with np.load(export_dir / "quantized_model.npz") as z:
        q = {k: z[k] for k in z.files}
    logits = integer_forward(pixels, q)
    pred = logits.argmax(1).astype(np.uint8)
    write_mem(vectors / "images.mem", pixels, 8)
    write_mem(vectors / "labels.mem", labels, 8)
    write_mem(vectors / "expected_logits.mem", logits, 32)
    write_mem(vectors / "expected_class.mem", pred, 8)
    return rows, labels, logits, pred, export_dir, out, vectors


def validate(run, iv, vvp):
    print(f"RTL_START {run}", flush=True); started = time.perf_counter()
    rows, labels, ref_logits, ref_pred, weights, out, vectors = prepare(run)
    binary = out / "dataset.vvp"
    cmd = [iv, "-g2012", "-s", "tb_dataset_batch", "-Ptb_dataset_batch.NTEST=200",
           f'-Ptb_dataset_batch.WEIGHTS_DIR="{weights.as_posix()}"',
           f'-Ptb_dataset_batch.VECTOR_DIR="{vectors.as_posix()}"', "-o", str(binary),
           *map(str, RTL), str(ROOT / "tb" / "tb_dataset_batch.sv")]
    comp = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "compile.log").write_text(comp.stdout, encoding="utf-8")
    if comp.returncode: raise RuntimeError(f"{run}: Icarus compile failed\n{comp.stdout}")
    sim = subprocess.run([vvp, str(binary)], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, timeout=1800)
    (out / "simulation.log").write_text(sim.stdout, encoding="utf-8")
    if sim.returncode or "PASS_DATASET_BATCH" not in sim.stdout:
        raise RuntimeError(f"{run}: RTL simulation failed\n{sim.stdout[-4000:]}")
    got = []
    for line in sim.stdout.splitlines():
        if line.startswith("RESULT,"):
            p = line.split(",")
            got.append([int(x) for x in p[1:]])
    if len(got) != 200: raise RuntimeError(f"{run}: received {len(got)}/200 results")
    got.sort(key=lambda x: x[0])
    rtl_pred = np.asarray([x[2] for x in got], dtype=np.uint8)
    rtl_logits = np.asarray([x[3:8] for x in got], dtype=np.int32)
    cycles = np.asarray([x[8] for x in got])
    reference_match = np.asarray([x[9] for x in got])
    exact_logits = bool(np.array_equal(rtl_logits, ref_logits))
    exact_class = bool(np.array_equal(rtl_pred, ref_pred))
    if not exact_logits or not exact_class or not np.all(reference_match == 1):
        raise RuntimeError(f"{run}: RTL and integer Python diverged")
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (labels, rtl_pred), 1)
    data = metrics(cm); elapsed = time.perf_counter() - started
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["index", "file", "true", "rtl_pred", "reference_pred", "logit0", "logit1", "logit2", "logit3", "logit4", "cycles", "reference_match"])
        for i, (row, true, pred, ref, lg, cy, mt) in enumerate(zip(rows, labels, rtl_pred, ref_pred, rtl_logits, cycles, reference_match)):
            w.writerow([i, row["file"], int(true), int(pred), int(ref), *map(int, lg), int(cy), int(mt)])
    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["true\\pred", *CLASSES]);
        for name, row in zip(CLASSES, cm): w.writerow([name, *map(int, row)])
    plot_matrix(cm, out / "confusion_matrix.png", f"RTL — {run} — teste reservado (n=200)")
    summary = {"run": run, "test_images": 200, "confusion_matrix": cm.tolist(), **data,
               "rtl_python_integer_logits_exact": exact_logits,
               "rtl_python_integer_class_exact": exact_class,
               "reference_matches": int(reference_match.sum()),
               "cycles_per_image_unique": sorted(set(map(int, cycles))),
               "elapsed_seconds": elapsed, "weights": str(weights.relative_to(ROOT))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RTL_DONE {run} accuracy={data['accuracy']:.6f} macro_f1={data['macro_f1']:.6f} exact=200/200 seconds={elapsed:.1f}", flush=True)
    return summary


def main():
    iv, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iv or not vvp: raise SystemExit("Icarus Verilog nao encontrado")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = [validate(run, iv, vvp) for run in RUNS]
    summary = {"status": "PASS", "protocol": "real Icarus Verilog on the same fixed 200-image test set",
               "runs": all_results, "mean_accuracy_folds": float(np.mean([x["accuracy"] for x in all_results[1:]])),
               "std_accuracy_folds": float(np.std([x["accuracy"] for x in all_results[1:]], ddof=1)),
               "all_rtl_python_integer_exact": all(x["reference_matches"] == 200 for x in all_results)}
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["run", "accuracy", "macro_precision", "macro_recall", "macro_f1", "exact_matches", "seconds"])
        for x in all_results: w.writerow([x["run"], x["accuracy"], x["macro_precision"], x["macro_recall"], x["macro_f1"], x["reference_matches"], x["elapsed_seconds"]])
    print(json.dumps({"status": "PASS", "mean_accuracy_folds": summary["mean_accuracy_folds"], "std_accuracy_folds": summary["std_accuracy_folds"], "all_exact": summary["all_rtl_python_integer_exact"]}), flush=True)


if __name__ == "__main__": main()
