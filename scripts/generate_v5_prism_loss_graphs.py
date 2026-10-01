"""Generate the matched Prism U-Net MSE and V5 MSE loss figure.

The source histories deliberately expose different validation fields: both
runs record chroma L1, while only V5 records aggregate validation chroma MSE.
The generated figure therefore compares the exact MSE training objective and
the common validation chroma-L1 metric without estimating missing values.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "research" / "reports" / "v5_prism_mse100"
PRISM_HISTORY = (
    ROOT
    / "models"
    / "experiments"
    / "prism"
    / "prism_coco_unet_mse100"
    / "prism_unet_mse"
    / "history.jsonl"
)
V5_HISTORY = ROOT / "models" / "experiments" / "v5_prism_mse100" / "history.jsonl"


def read_history(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, 101)):
        raise ValueError(f"Expected epochs 1--100 in {path}, found {epochs[:3]}...{epochs[-3:]}")
    return rows


def metric_rows(prism: list[dict], v5: list[dict]) -> list[dict[str, float | int]]:
    return [
        {
            "epoch": epoch,
            "prism_train_mse": prism_row["training"]["reconstruction"],
            "v5_train_mse": v5_row["training"]["chroma_mse"],
            "prism_train_l1": prism_row["training"]["chroma_l1"],
            "v5_train_l1": v5_row["training"]["chroma_l1"],
            "prism_validation_l1": prism_row["validation"]["chroma_l1"],
            "v5_validation_l1": v5_row["validation"]["chroma_l1"],
            "v5_validation_mse": v5_row["validation"]["chroma_mse"],
        }
        for epoch, (prism_row, v5_row) in enumerate(zip(prism, v5, strict=True), start=1)
    ]


def write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: value if isinstance(value, int) else f"{value:.12g}"
                    for key, value in row.items()
                }
            )


def write_tex(rows: list[dict[str, float | int]], path: Path) -> None:
    last = rows[-1]
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[paperwidth=20cm,paperheight=12.6cm,margin=0.55cm]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\pgfplotsset{{compat=1.18}}
\pagestyle{{empty}}

\definecolor{{prismblue}}{{RGB}}{{31,119,180}}
\definecolor{{v5orange}}{{RGB}}{{230,126,34}}

\begin{{document}}
\begin{{center}}
  {{\Large\bfseries Matched Prism U-Net and V5 MSE training}}\\[1mm]
  {{\small Same COCO split, degradations, crops, optimizer, schedule, seed, and 100-epoch horizon}}
\end{{center}}

\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 1, horizontal sep=1.55cm}},
  width=8.95cm,
  height=8.25cm,
  xmin=1,
  xmax=100,
  xtick={{1,20,40,60,80,100}},
  xlabel={{Epoch}},
  grid=both,
  major grid style={{gray!28}},
  minor grid style={{gray!12}},
  axis line style={{black!65}},
  tick label style={{font=\small}},
  label style={{font=\small}},
  title style={{font=\small\bfseries, align=center}},
  legend style={{
    font=\small,
    draw=none,
    fill=white,
    fill opacity=0.88,
    text opacity=1,
    cells={{anchor=west}},
  }},
  no markers,
]

\nextgroupplot[
  title={{(a) Training objective}},
  ylabel={{Chroma MSE}},
  ymode=log,
  log basis y=10,
  ymin=5e-5,
  ymax=5e-4,
  legend pos=north east,
]
\addplot+[prismblue, very thick]
  table[x=epoch,y=prism_train_mse,col sep=comma] {{loss_curves.csv}};
\addlegendentry{{Prism U-Net MSE}}
\addplot+[v5orange, very thick, dashed]
  table[x=epoch,y=v5_train_mse,col sep=comma] {{loss_curves.csv}};
\addlegendentry{{V5 MSE}}

\nextgroupplot[
  title={{(b) Common held-out metric}},
  ylabel={{Validation chroma L1 (0--1)}},
  ymin=0.0037,
  ymax=0.0082,
  ytick={{0.004,0.005,0.006,0.007,0.008}},
  scaled y ticks=false,
  yticklabel style={{/pgf/number format/fixed,/pgf/number format/precision=3,font=\small}},
  legend pos=north east,
]
\addplot+[prismblue, very thick]
  table[x=epoch,y=prism_validation_l1,col sep=comma] {{loss_curves.csv}};
\addlegendentry{{Prism U-Net MSE}}
\addplot+[v5orange, very thick, dashed]
  table[x=epoch,y=v5_validation_l1,col sep=comma] {{loss_curves.csv}};
\addlegendentry{{V5 MSE}}

\end{{groupplot}}
\end{{tikzpicture}}

\vspace{{-1mm}}
\begin{{center}}
\small
Epoch 100: training MSE $={last['prism_train_mse']:.3e}$ (Prism) and
$={last['v5_train_mse']:.3e}$ (V5); validation chroma L1
$={last['prism_validation_l1']:.6f}$ (Prism) and
$={last['v5_validation_l1']:.6f}$ (V5).

\footnotesize\itshape
The Prism log does not contain aggregate validation MSE, so panel (b) uses the
held-out chroma L1 metric recorded by both runs; no missing metric is inferred.
\end{{center}}
\end{{document}}
"""
    path.write_text(tex, encoding="utf-8", newline="\n")


def main() -> None:
    prism = read_history(PRISM_HISTORY)
    v5 = read_history(V5_HISTORY)
    rows = metric_rows(prism, v5)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, REPORT_DIR / "loss_curves.csv")
    write_tex(rows, REPORT_DIR / "loss_curves.tex")
    print(f"Wrote {len(rows)} aligned epochs to {REPORT_DIR / 'loss_curves.csv'}")
    print(f"Wrote standalone PGFPlots source to {REPORT_DIR / 'loss_curves.tex'}")


if __name__ == "__main__":
    main()
