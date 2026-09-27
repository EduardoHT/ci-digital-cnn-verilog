"""Agrega predicoes HDL, calcula matrizes/metricas e gera CSV, SVG e relatorio."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

CLASSES = ["circulo", "quadrado", "triangulo", "cruz", "estrela"]
SPLITS = ["train", "val", "test"]


def confusion(rows: list[dict[str, str]]) -> list[list[int]]:
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    for row in rows:
        matrix[int(row["true"])][int(row["pred"])] += 1
    return matrix


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metrics(matrix: list[list[int]]) -> dict:
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(5))
    per_class = []
    for i, name in enumerate(CLASSES):
        support = sum(matrix[i])
        predicted = sum(matrix[r][i] for r in range(5))
        tp = matrix[i][i]
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, support)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class.append({"class": name, "support": support, "predicted": predicted, "tp": tp,
                          "precision": precision, "recall": recall, "f1": f1})
    macro = {k: sum(item[k] for item in per_class) / 5 for k in ["precision", "recall", "f1"]}
    weighted = {k: safe_div(sum(item[k] * item["support"] for item in per_class), total)
                for k in ["precision", "recall", "f1"]}
    return {"total": total, "correct": correct, "accuracy": safe_div(correct, total),
            "per_class": per_class, "macro": macro, "weighted": weighted,
            "zero_division": "precision, recall ou F1 com denominador zero recebem 0.0"}


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASSES, "total"])
        for name, row in zip(CLASSES, matrix, strict=True):
            writer.writerow([name, *row, sum(row)])
        writer.writerow(["total", *[sum(matrix[r][c] for r in range(5)) for c in range(5)], sum(map(sum, matrix))])


def write_metrics(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "class", "support", "predicted", "tp", "precision", "recall", "f1"])
        for item in data["per_class"]:
            writer.writerow(["class", item["class"], item["support"], item["predicted"], item["tp"],
                             f'{item["precision"]:.9f}', f'{item["recall"]:.9f}', f'{item["f1"]:.9f}'])
        writer.writerow(["macro", "ALL", data["total"], "", data["correct"],
                         f'{data["macro"]["precision"]:.9f}', f'{data["macro"]["recall"]:.9f}', f'{data["macro"]["f1"]:.9f}'])
        writer.writerow(["weighted", "ALL", data["total"], "", data["correct"],
                         f'{data["weighted"]["precision"]:.9f}', f'{data["weighted"]["recall"]:.9f}', f'{data["weighted"]["f1"]:.9f}'])
        writer.writerow(["accuracy", "ALL", data["total"], "", data["correct"], "", "", f'{data["accuracy"]:.9f}'])


def matrix_markdown(matrix: list[list[int]]) -> str:
    lines = ["| verdadeira \\ prevista | " + " | ".join(CLASSES) + " | total |",
             "| --- | " + " | ".join(["---:"] * 6) + " |"]
    for name, row in zip(CLASSES, matrix, strict=True):
        lines.append("| " + name + " | " + " | ".join(map(str, [*row, sum(row)])) + " |")
    return "\n".join(lines)


def write_svg(path: Path, matrix: list[list[int]], title: str) -> None:
    max_value = max(max(row) for row in matrix) or 1
    cell, left, top = 100, 150, 90
    width, height = left + 5 * cell + 30, top + 5 * cell + 80
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{title}</text>',
             f'<text x="{left+250}" y="58" text-anchor="middle" font-family="sans-serif" font-size="14">classe prevista</text>']
    for i, name in enumerate(CLASSES):
        parts.append(f'<text x="{left+i*cell+cell/2}" y="{top-10}" text-anchor="middle" font-family="sans-serif" font-size="12">{name}</text>')
        parts.append(f'<text x="{left-12}" y="{top+i*cell+cell/2+4}" text-anchor="end" font-family="sans-serif" font-size="12">{name}</text>')
    parts.append(f'<text transform="translate(20 {top+250}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">classe verdadeira</text>')
    for r in range(5):
        for c in range(5):
            value = matrix[r][c]
            shade = int(245 - 175 * value / max_value)
            fill = f'rgb({shade},{shade+5},{255})'
            parts.append(f'<rect x="{left+c*cell}" y="{top+r*cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="#666"/>')
            parts.append(f'<text x="{left+c*cell+cell/2}" y="{top+r*cell+cell/2+7}" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{value}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--started-unix", type=int, default=0)
    args = parser.parse_args()
    files = sorted(args.input.glob("predictions_shard_*.csv"))
    if len(files) != 10:
        raise RuntimeError(f"Esperados 10 lotes, encontrados {len(files)}")
    rows = []
    for file in files:
        rows.extend(csv.DictReader(file.open(encoding="utf-8", newline="")))
    rows.sort(key=lambda row: int(row["global_index"]))
    indices = [int(row["global_index"]) for row in rows]
    if indices != list(range(2000)):
        raise RuntimeError("Indices globais ausentes, duplicados ou fora de ordem")
    if any(int(row["reference_match"]) != 1 or int(row["cycles"]) != 206520 for row in rows):
        raise RuntimeError("Falha de referencia ou contagem de ciclos")
    split_counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    if split_counts != {"train": 1400, "val": 300, "test": 300}:
        raise RuntimeError(f"Contagem por split invalida: {split_counts}")
    true_counts = [sum(int(row["true"]) == i for row in rows) for i in range(5)]
    if true_counts != [400] * 5:
        raise RuntimeError(f"Contagem por classe invalida: {true_counts}")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fields = ["global_index", "split", "id", "file", "true", "pred", "logit0", "logit1", "logit2", "logit3", "logit4", "cycles", "reference_match"]
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    scopes = {"total": rows, **{split: [row for row in rows if row["split"] == split] for split in SPLITS}}
    summary = {"status": "PASS", "weights": "exports/demo", "classes": CLASSES,
               "rows_true_columns_predicted": True, "split_counts": split_counts,
               "unique_images": len({row["file"] for row in rows}), "reference_matches": 2000,
               "cycles_per_image": 206520, "coverage": {
                   "all_2000": "logits, argmax, labels, image hashes, busy/done, cycles, timeout",
                   "full_internal_maps": "10 demo vectors in tb_cnn.sv, 9102 exact values per vector"},
               "scopes": {}}
    report = ["# Campanha RTL completa — pesos sintéticos — 2.000 imagens", "",
              "Execução real do `tiny_cnn` em Icarus Verilog. Python foi usado apenas para preparar a referência inteira e agregar as linhas emitidas pelo simulador.", ""]
    run_id = os.getenv("GITHUB_RUN_ID", "")
    if run_id:
        run_url = f"https://github.com/EduardoHT/ci-digital-cnn-verilog/actions/runs/{run_id}"
        summary["run_url"] = run_url
        report += [f"- Execução: [{run_id}]({run_url})"]
    duration = int(time.time()) - args.started_unix if args.started_unix else None
    summary["duration_seconds_to_aggregation"] = duration
    report += [f"- Imagens processadas: **{len(rows)}**", f"- Comparações RTL × referência aprovadas: **2000/2000**",
               f"- Duração até a agregação: **{duration} s**" if duration is not None else "- Duração: consulte a execução", ""]
    for name, scope_rows in scopes.items():
        matrix = confusion(scope_rows)
        data = metrics(matrix)
        summary["scopes"][name] = {"confusion": matrix, **data}
        write_matrix(out / f"confusion_{name}.csv", matrix)
        write_metrics(out / f"metrics_{name}.csv", data)
        write_svg(out / f"confusion_{name}.svg", matrix, f"Matriz de confusão — {name} — pesos sintéticos")
        report += [f"## {name}", "", matrix_markdown(matrix), "",
                   f"Acurácia funcional de distribuição: **{data['correct']}/{data['total']} = {data['accuracy']:.6f}**. ",
                   f"Macro F1: **{data['macro']['f1']:.6f}**. Weighted F1: **{data['weighted']['f1']:.6f}**.", ""]
    report += ["## Interpretação", "",
               "Estas matrizes descrevem a distribuição de saídas dos pesos sintéticos e validam o pipeline em escala. Não são métricas de generalização. Apenas a matriz `test` usa o split reservado, e mesmo ela não mede um modelo treinado.", "",
               "Divisão por zero: precision, recall ou F1 recebem `0.0` quando o denominador correspondente é zero.", "",
               "Cobertura: todas as 2.000 imagens tiveram logits, classe, protocolo e ciclos comparados; os mapas internos completos foram comparados nos dez vetores demo pelo testbench detalhado.", ""]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(rows), "split_counts": split_counts, "duration": duration}), flush=True)


if __name__ == "__main__":
    main()
