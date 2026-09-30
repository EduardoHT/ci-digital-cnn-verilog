"""Baseline 70/20/10 and paired 5-fold evaluation for the 28x28 ShapeCNN.

The same stratified 10% test set is held out for every run.  Cross-validation
uses only the remaining 90%.  Every output is written below runs/cv_70_20_10.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from python.shapes import (
    ARCHITECTURE,
    CLASSES,
    IMAGE_SIZE,
    SEED,
    ShapeCNN,
    classification_metrics,
    get_loader,
    predict_float,
    seed_everything,
)
from python.quantization import run_export

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
OUT = ROOT / "runs" / "cv_70_20_10"
EXPORTS = ROOT / "exports" / "cv_70_20_10"
EPOCHS = 40
PATIENCE = 10
BATCH = 64
LR = 0.002


def load_all():
    rows = list(csv.DictReader((DATASET / "manifest.csv").open(encoding="utf-8")))
    if len(rows) != 2000:
        raise RuntimeError(f"Expected 2000 manifest rows, found {len(rows)}")
    hashes = [r["sha256_pixels"] for r in rows]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("Duplicate or variant group detected by pixel hash")
    pixels, labels, paths = [], [], []
    for row in rows:
        path = DATASET / row["file"]
        with Image.open(path) as im:
            if im.mode != "L" or im.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"Invalid image: {path} ({im.mode}, {im.size})")
            pixels.append(np.asarray(im, dtype=np.uint8).copy())
        labels.append(int(row["label"]))
        paths.append(row["file"])
    return np.stack(pixels)[:, None], np.asarray(labels), np.asarray(paths), rows


def split_indices(labels):
    rng = np.random.default_rng(SEED)
    test, dev = [], []
    for cls in range(len(CLASSES)):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        test.extend(idx[:40])
        dev.extend(idx[40:])
    test = np.asarray(sorted(test))
    dev = np.asarray(dev)
    baseline_train, baseline_val = [], []
    folds = [[] for _ in range(5)]
    for cls in range(len(CLASSES)):
        idx = dev[labels[dev] == cls].copy()
        rng.shuffle(idx)
        baseline_train.extend(idx[:280])
        baseline_val.extend(idx[280:])
        for fold in range(5):
            folds[fold].extend(idx[fold * 72:(fold + 1) * 72])
    return test, np.asarray(sorted(baseline_train)), np.asarray(sorted(baseline_val)), [np.asarray(sorted(x)) for x in folds], np.asarray(sorted(dev))


def evaluate(model, pixels, labels):
    logits = predict_float(model, pixels, "cpu")
    return classification_metrics(labels, logits.argmax(1)), logits


def train_one(name, train_idx, val_idx, test_idx, pixels, labels, paths, rows):
    run_dir = OUT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = SEED + sum(ord(c) for c in name)
    seed_everything(seed)
    model = ShapeCNN().cpu()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = get_loader(pixels[train_idx], labels[train_idx], BATCH, True, seed)
    val_loader = get_loader(pixels[val_idx], labels[val_idx], 128)
    history, best, best_epoch, stalled = [], (-1.0, float("inf")), 0, 0
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0.0; correct = 0; count = 0
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(x); loss = loss_fn(logits, y)
            loss.backward(); opt.step()
            total_loss += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum()); count += len(y)
        model.eval(); val_loss = 0.0; val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                val_loss += float(loss_fn(logits, y).item()) * len(y)
                val_correct += int((logits.argmax(1) == y).sum())
        va, vl = val_correct / len(val_idx), val_loss / len(val_idx)
        rec = {"epoch": epoch, "train_loss": total_loss / count,
               "train_accuracy": correct / count, "val_loss": vl,
               "val_accuracy": va}
        history.append(rec)
        print(f"{name} epoch={epoch:02d} train={rec['train_accuracy']:.4f} val={va:.4f} val_loss={vl:.4f}", flush=True)
        if va > best[0] or (va == best[0] and vl < best[1]):
            best, best_epoch, stalled = (va, vl), epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            stalled += 1
        if stalled >= PATIENCE:
            break
    elapsed = time.perf_counter() - start
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    val_metrics, _ = evaluate(model, pixels[val_idx], labels[val_idx])
    test_metrics, test_logits = evaluate(model, pixels[test_idx], labels[test_idx])
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    manifest_rows = []
    membership = {int(i): "train" for i in train_idx}
    membership.update({int(i): "val" for i in val_idx})
    membership.update({int(i): "test" for i in test_idx})
    for i, row in enumerate(rows):
        if i in membership:
            manifest_rows.append({"split": membership[i], **row})
    with (run_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0])); w.writeheader(); w.writerows(manifest_rows)
    np.save(run_dir / "test_logits.npy", test_logits)
    report = {"name": name, "architecture": ARCHITECTURE, "image_size": [28, 28],
              "parameters": sum(p.numel() for p in model.parameters()), "seed": seed,
              "train_samples": len(train_idx), "val_samples": len(val_idx),
              "test_samples": len(test_idx), "epochs_run": len(history),
              "best_epoch": best_epoch, "elapsed_seconds": elapsed,
              "validation": val_metrics, "heldout_test": test_metrics}
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Materialize only the data needed by the unchanged RTL export pipeline.
    exp_data = run_dir / "export_dataset"
    for split, idxs in (("train", train_idx), ("test", test_idx)):
        for i in idxs:
            dst = exp_data / split / CLASSES[int(labels[i])] / Path(paths[i]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(DATASET / paths[i], dst)
    export_dir = EXPORTS / name
    run_export(run_dir / "best.pt", export_dir, exp_data, per_class=3, overwrite=True)
    return report


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    pixels, labels, paths, rows = load_all()
    test, btrain, bval, folds, dev = split_indices(labels)
    OUT.mkdir(parents=True, exist_ok=True); EXPORTS.mkdir(parents=True, exist_ok=True)
    runs = [("baseline", btrain, bval)]
    for fold, val in enumerate(folds, 1):
        train = np.setdiff1d(dev, val, assume_unique=True)
        runs.append((f"fold_{fold}", train, val))
    reports = []
    for name, train, val in runs:
        reports.append(train_one(name, train, val, test, pixels, labels, paths, rows))
    summary = {
        "protocol": "stratified baseline 70/20/10 plus 5-fold on the 90% development set",
        "fixed_test_samples": int(len(test)), "runs": reports,
        "fold_test_accuracy_mean": float(np.mean([r["heldout_test"]["accuracy"] for r in reports[1:]])),
        "fold_test_accuracy_std": float(np.std([r["heldout_test"]["accuracy"] for r in reports[1:]], ddof=1)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["run", "epochs", "best_epoch", "val_accuracy", "test_accuracy", "test_macro_f1", "seconds"])
        for r in reports:
            w.writerow([r["name"], r["epochs_run"], r["best_epoch"], r["validation"]["accuracy"], r["heldout_test"]["accuracy"], r["heldout_test"]["macro_f1"], r["elapsed_seconds"]])
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
