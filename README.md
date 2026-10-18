# CoverPIR

Implementation of **Evidence-hypergraph layout for fixed-budget private multi-hop retrieval**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data

Each corpus maps chunk IDs to text. Train queries provide question and complete support IDs; test query support/answer is used only for scoring. Public fields fix B reads, padded block bytes L, block count and total storage R. [HotpotQA](https://hotpotqa.github.io/) supplies sentence-level supports for imported corpora.

The bundled generator writes synthetic finite-model scenarios as JSONL. Fix the seed and keep the generated input for paired comparisons.

```bash
python -m coverpir prepare --count 200 --seed 17 --output data/generated.jsonl
```

`configs/finite_reader.json` joins the generated City→Person→District→Region facts using only retrieved records. Imported natural-language corpora require a separately configured local answerer; omitting `--answerer-config` produces retrieval-only metrics.

## Execution and evaluation

```bash
python -m coverpir run --data data/generated.jsonl --answerer-config configs/finite_reader.json --output runs/main
```

## Comparisons and ablations

```bash
python -m coverpir suite --data data/generated.jsonl --answerer-config configs/finite_reader.json --output runs/comparison
```

Use `--methods` to select: `coverpir`, `semantic`, `random_copies`, `pairwise`, `no_copies`, `gold_routing`, `no_cost`.

## Outputs

`runs/main/coverpir/records.jsonl` stores per-case results; `metrics.json` stores aggregate measurements. Comparison outputs use one directory per method.

```bash
python -m coverpir plot --inputs runs/comparison/coverpir --output figures
```

Plotting writes `figures/summary.csv` and metric PNGs. Statistical intervals can be computed from a retained record file:

```bash
python -m coverpir.statistics --records runs/main/coverpir/records.jsonl --output results/intervals.json
```
