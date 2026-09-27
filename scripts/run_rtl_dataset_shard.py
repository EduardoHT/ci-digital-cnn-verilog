"""Prepara, simula e registra um lote do dataset usando o tiny_cnn real."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTL = [ROOT / "rtl" / name for name in ["mac_u8s8.sv", "relu_requant.sv", "max4_u8.sv", "tiny_cnn.sv"]]


def run(cmd: list[str], *, log: Path | None = None, timeout: int = 1200) -> str:
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if log:
        log.write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(f"Comando falhou ({result.returncode})")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    iv, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not iv or not vvp:
        raise SystemExit("Icarus Verilog nao encontrado")

    build = ROOT / "build" / "full_dataset" / f"shard_{args.shard_index}"
    build.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        str(ROOT / "scripts" / "prepare_rtl_dataset_shard.py"),
        "--shard-index", str(args.shard_index),
        "--shard-count", str(args.shard_count),
        "--out", str(build),
    ])
    summary = json.loads((build / "summary.json").read_text(encoding="utf-8"))
    ntest = int(summary["n_vectors"])
    binary = build / "dataset.vvp"
    compile_cmd = [
        iv, "-g2012", "-s", "tb_dataset_batch",
        f"-Ptb_dataset_batch.NTEST={ntest}",
        f'-Ptb_dataset_batch.WEIGHTS_DIR="{(ROOT / "exports" / "demo").as_posix()}"',
        f'-Ptb_dataset_batch.VECTOR_DIR="{build.as_posix()}"',
        "-o", str(binary), *map(str, RTL), str(ROOT / "tb" / "tb_dataset_batch.sv"),
    ]
    run(compile_cmd, log=build / "compile.log")
    output = run([vvp, str(binary)], log=build / "simulation.log", timeout=1200)
    result_rows = []
    for line in output.splitlines():
        if not line.startswith("RESULT,"):
            continue
        parts = line.split(",")
        if len(parts) != 11:
            raise RuntimeError(f"Linha RESULT invalida: {line}")
        result_rows.append({
            "local_index": int(parts[1]), "true_hdl": int(parts[2]), "pred": int(parts[3]),
            "logit0": int(parts[4]), "logit1": int(parts[5]), "logit2": int(parts[6]),
            "logit3": int(parts[7]), "logit4": int(parts[8]), "cycles": int(parts[9]),
            "reference_match": int(parts[10]),
        })
    if "PASS_DATASET_BATCH" not in output or len(result_rows) != ntest:
        raise RuntimeError(f"Lote incompleto: {len(result_rows)}/{ntest}")
    by_local = {row["local_index"]: row for row in result_rows}
    if set(by_local) != set(range(ntest)):
        raise RuntimeError("Indices locais ausentes ou duplicados")

    samples = list(csv.DictReader((build / "samples.csv").open(encoding="utf-8", newline="")))
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / f"predictions_shard_{args.shard_index:02d}.csv"
    fields = ["global_index", "split", "id", "file", "true", "pred", "logit0", "logit1", "logit2", "logit3", "logit4", "cycles", "reference_match"]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            local = int(sample["local_index"])
            got = by_local[local]
            if got["true_hdl"] != int(sample["true"]):
                raise RuntimeError(f"Rotulo HDL divergente no indice {local}")
            if got["pred"] != int(sample["reference_pred"]) or got["reference_match"] != 1:
                raise RuntimeError(f"Referencia divergente no indice {local}")
            writer.writerow({name: sample[name] if name in sample else got[name] for name in fields})
    (args.out / f"summary_shard_{args.shard_index:02d}.json").write_text(
        json.dumps({**summary, "status": "PASS", "results": len(result_rows)}, indent=2), encoding="utf-8"
    )
    print(f"PASS_SHARD index={args.shard_index} vectors={len(result_rows)}", flush=True)


if __name__ == "__main__":
    main()
