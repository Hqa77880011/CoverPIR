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

Deployed and oracle evidence coverage, support recall, configured client-local multi-hop answer F1/exact match, online bytes, logical reads and padded storage. Without an answerer, answer metrics are unavailable rather than replaced with a query-overlap sentence score.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m coverpir plot --inputs runs/comparison/coverpir --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

The single-server PIR backend uses 2048-bit Paillier encrypted one-hot selectors and homomorphic packed-record responses; key generation and computation are included in local timing. It is a correctness-oriented backend, with different cost from high-throughput PIR primitives. Answering is a distinct trusted-client step over the recovered text, with no test answer or supporting-fact labels supplied to it. JSON serialization includes IDs and text before zero padding; one dummy block is reserved.

## Multi-hop answering from recovered evidence

Choose one fixed answer model and generation configuration for every layout. The answering stage starts only after exactly B private reads have completed. It receives the question and recovered `{id,text}` records, and may combine all required hops. A command or model failure propagates; there is no word-overlap fallback. Answer EM/F1 lowercases, removes punctuation and English articles, and normalizes whitespace; mismatched yes/no/noanswer answers score zero if either side is special. Cited-support precision/recall/F1 is reported separately. The answerer configuration and local answer latency are retained in results. An offline Transformers model is loaded once before per-case timing; measured answer latency covers its generation, not cold-start loading.

For an existing local instruction model, install `python -m pip install -e ".[local]"` and save an answerer JSON file:

```json
{
  "backend": "transformers",
  "model_path": "../models/local-instruction-model",
  "device": "cpu",
  "max_new_tokens": 128,
  "context_tokens": 4096
}
```

`model_path` resolves relative to this JSON file and must contain already downloaded weights/tokenizer with a chat template. Both load with `local_files_only=true`; no API key, remote service, model download or remote custom code is used. Generation is deterministic. If all recovered evidence does not fit the registered context window, the run fails explicitly; choose compatible block/read budgets rather than silently dropping a hop.

```bash
python -m coverpir run --data data/workload.jsonl --answerer-config configs/answerer.json --output runs/answered
python -m coverpir suite --data data/workload.jsonl --answerer-config configs/answerer.json --output runs/answered_comparison
```

A trusted local QA executable can instead use `{"backend":"command","command":["python","local_multihop_reader.py"],"cwd":"..","timeout_seconds":120}`. `cwd` resolves relative to the JSON config. The process receives exactly one JSON object on stdin:

```json
{"protocol":"coverpir.answer.v1","question":"In which region was the governor of City 7 born?","evidence":[{"id":"21","text":"City 7 is governed by Person 7."},{"id":"22","text":"Person 7 was born in District 7."},{"id":"23","text":"District 7 belongs to Region 7."}]}
```

It must return one stdout JSON object, for example `{"answer":"Region 7","support_ids":["21","22","23"]}`; this example illustrates the protocol and is not a built-in research answerer. `support_ids` must be recovered chunk IDs. An insufficient-evidence reply is `{"answer":"","support_ids":[]}`. Gold supports and answers remain confined to evaluation/oracle routing. Use local processes because forwarding the private question to an external answering service changes the paper's trusted-client boundary.

Running without `--answerer-config` is explicitly retrieval-only: `answer_f1`, `answer_exact_match` and answer-support metrics are null. Existing historical query-overlap answer scores are not comparable to this actual QA interface. HotpotQA fullwiki corpus/chunk/support preparation and a trained local reader remain required for the paper-scale evaluation; the supplied finite generator is a protocol workload.


## Uncertainty and paired comparisons

```bash
python -m coverpir.statistics --records runs/main/coverpir/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.

Layout search evaluates byte-feasible moves, swaps and copies. Optional validation_queries supplies a held-out public checkpoint-selection set while the router remains fitted to train_queries.

The declared offline-cost proxy prices extra chunk placements via replica_cost_weight; no_cost removes this penalty. Physical padded database dimensions remain fixed across layouts.

## Fixed reproduction setting

Finite evidence-hypergraph retrieval with the actual Paillier correctness backend and fixed read budget. QA is unavailable until a real reader is explicitly supplied.

The checked-in entry point fixes data seeds, methods, budgets and output paths. Inspect its commands without starting any computation:

```bash
python reproduce.py --plan
```

Remove `--plan` to execute the registered pipeline in a fresh output directory. The entry point calls the existing implementation; it does not insert paper results. Installation uses the package instructions above.
