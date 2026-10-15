# CoverPIR

Implementation of **Evidence-hypergraph layout for fixed-budget private multi-hop retrieval**.

Fits a local support-set router, searches byte-feasible replicated layouts and executes exactly B single-server Paillier private reads. Gold routing is confined to the diagnostic method.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/CoverPIR.git
cd CoverPIR
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

Each corpus maps chunk IDs to text. Train queries provide question and complete support IDs; test query support/answer is used only for scoring. Public fields fix B reads, padded block bytes L, block count and total storage R. [HotpotQA](https://hotpotqa.github.io/) supplies sentence-level supports for imported corpora.

```bash
python -m coverpir prepare --count 200 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m coverpir run --data data/workload.jsonl --output runs/main
python -m coverpir suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. coverpir optimizes deployed complete-edge coverage; semantic uses capacity-aware TF-IDF cluster packing; random_copies accepts feasible copies; pairwise rewards pair co-occurrence; no_copies pads the same physical dimensions; gold_routing uses test supports as an oracle.

## Evaluation and visualization

Deployed and oracle evidence coverage, support recall, local extractive answer F1/exact match, online bytes, logical reads and padded storage.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m coverpir plot --inputs runs/comparison/coverpir --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

The single-server PIR backend uses 2048-bit Paillier encrypted one-hot selectors and homomorphic packed-record responses; key generation and computation are included in local timing. It is a correctness-oriented backend, with different cost from high-throughput PIR primitives. The local answerer selects a retrieved sentence by query-token overlap and never reads the gold answer. JSON serialization includes IDs and text before zero padding; one dummy block is reserved.


## Uncertainty and paired comparisons

```bash
python -m coverpir.statistics --records runs/main/coverpir/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.

Layout search evaluates byte-feasible moves, swaps and copies. Optional validation_queries supplies a held-out public checkpoint-selection set while the router remains fitted to train_queries.

The declared offline-cost proxy prices extra chunk placements via replica_cost_weight; no_cost removes this penalty. Physical padded database dimensions remain fixed across layouts.
