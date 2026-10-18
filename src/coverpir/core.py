import itertools
import json
import re
import string
import time
from collections import Counter

import numpy as np
from phe import paillier
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


METHODS = [
    "coverpir",
    "semantic",
    "random_copies",
    "pairwise",
    "no_copies",
    "gold_routing",
    "no_cost",
]


def serialize(block, chunks, size):
    payload = json.dumps(
        [{"id": i, "text": chunks[i]} for i in sorted(block)],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    if len(payload) > size:
        raise ValueError("Serialized block exceeds physical capacity")
    return payload + bytes(size - len(payload))


def oracle_cover(layout, support, budget):
    choices = []
    for indices in itertools.combinations(range(len(layout)), min(budget, len(layout))):
        covered = set().union(*(layout[i] for i in indices))
        choices.append((len(covered & set(support)), indices))
    return max(choices, default=(0, ()), key=lambda x: (x[0], tuple(-i for i in x[1])))[1]


def router(layout, chunks, train, budget):
    vectorizer = TfidfVectorizer(max_features=128, ngram_range=(1, 2))
    vectorizer.fit(list(chunks.values()) + [q["question"] for q in train])
    x = vectorizer.transform([q["question"] for q in train]).toarray()
    x = np.column_stack([np.ones(len(x)), x])
    y = np.zeros((len(train), len(layout)))
    for row, query in enumerate(train):
        y[row, list(oracle_cover(layout, query["support"], budget))] = 1
    positive = max(1.0, (y.size - y.sum()) / max(1.0, y.sum()))

    def objective(flat):
        w = flat.reshape(x.shape[1], len(layout))
        logits = x @ w
        probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        weights = 1 + (positive - 1) * y
        loss = np.mean(weights * (np.logaddexp(0, logits) - y * logits)) + 1e-4 * (w**2).sum()
        gradient = x.T @ (weights * (probability - y)) / y.size + 2e-4 * w
        return loss, gradient.ravel()

    fitted = minimize(
        objective,
        np.zeros(x.shape[1] * len(layout)),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 100},
    )
    weights = fitted.x.reshape(x.shape[1], len(layout))

    def route(question):
        features = np.r_[1.0, vectorizer.transform([question]).toarray()[0]]
        scores = features @ weights
        return np.argsort(-scores, kind="stable")[:budget].tolist()

    return route


def layout_score(layout, chunks, queries, budget, method, validation=None):
    if method == "pairwise":
        return sum(
            any({a, b} <= block for block in layout)
            for q in queries
            for a, b in itertools.combinations(q["support"], 2)
        )
    route = router(layout, chunks, queries, budget)
    return sum(
        set(q["support"]) <= set().union(*(layout[i] for i in route(q["question"])))
        for q in (validation if validation is not None else queries)
    )


def build_layout(case, method):
    chunks, size, slots = case["chunks"], case["block_bytes"], case["block_slots"]
    if slots * size > case["storage_budget"]:
        raise ValueError("Physical padded storage exceeds R")
    layout = [set() for _ in range(slots)]
    cursor = 0
    names = sorted(chunks)
    embeddings = TfidfVectorizer().fit_transform([chunks[name] for name in names])
    estimated_blocks = max(
        1,
        min(
            slots - 1, int(np.ceil(sum(len(text.encode()) + 32 for text in chunks.values()) / size))
        ),
    )
    clusters = KMeans(
        n_clusters=estimated_blocks, random_state=case["seed"], n_init=10
    ).fit_predict(embeddings)
    ordered = sorted(zip(clusters, names))
    for _, chunk in ordered:
        while cursor < slots - 1:
            try:
                serialize(layout[cursor] | {chunk}, chunks, size)
                break
            except ValueError:
                cursor += 1
        if cursor >= slots - 1:
            raise ValueError("Insufficient base storage plus reserved dummy block")
        layout[cursor].add(chunk)
    if method in ("semantic", "no_copies"):
        return layout

    def objective(candidate):
        coverage = layout_score(
            candidate,
            chunks,
            case["train_queries"],
            case["read_budget"],
            method,
            case.get("validation_queries"),
        )
        replicas = sum(map(len, candidate)) - len(chunks)
        penalty = (
            0
            if method == "no_cost"
            else case.get("replica_cost_weight", 0.05) * replicas / len(chunks)
        )
        return coverage - penalty

    value = objective(layout)
    rng = np.random.default_rng(case["seed"])
    for _ in range(case.get("search_steps", 10)):
        chunk = str(rng.choice(sorted(chunks)))
        destination = int(rng.integers(slots - 1))
        if chunk in layout[destination]:
            continue
        candidate = [set(block) for block in layout]
        operation = "copy" if method == "random_copies" else rng.choice(["copy", "move", "swap"])
        origin = int(rng.choice([i for i, block in enumerate(candidate) if chunk in block]))
        if operation == "move":
            candidate[origin].remove(chunk)
        elif operation == "swap" and candidate[destination]:
            other = str(rng.choice(sorted(candidate[destination])))
            candidate[destination].remove(other)
            candidate[origin].remove(chunk)
            candidate[origin].add(other)
        candidate[destination].add(chunk)
        try:
            for block in candidate:
                serialize(block, chunks, size)
        except ValueError:
            continue
        proposed = objective(candidate)
        if proposed > value or method == "random_copies":
            layout, value = candidate, proposed
    return layout


def private_read(database, index, public_key, private_key):
    query = [public_key.encrypt(int(i == index)) for i in range(len(database))]
    chunk_bytes = (public_key.max_int.bit_length() - 1) // 8
    results = []
    for start in range(0, len(database[0]), chunk_bytes):
        segment = [
            int.from_bytes(record[start : start + chunk_bytes], "big") for record in database
        ]
        answer = sum(
            (encrypted * value for encrypted, value in zip(query, segment)), public_key.encrypt(0)
        )
        width = min(chunk_bytes, len(database[0]) - start)
        results.append(int(private_key.decrypt(answer)).to_bytes(width, "big"))
    ciphertext_bytes = (public_key.nsquare.bit_length() + 7) // 8
    return b"".join(results), len(query) * ciphertext_bytes + len(results) * ciphertext_bytes


def tokens(text):
    normalized = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", normalized).split()


def generate(rng, count):
    for index in range(count):
        people, districts, regions = [list(range(8)) for _ in range(3)]
        rng.shuffle(people)
        rng.shuffle(districts)
        rng.shuffle(regions)
        chunks = {}
        for i in range(8):
            chunks[str(3 * i)] = f"City {i} is governed by Person {people[i]}."
            chunks[str(3 * i + 1)] = f"Person {i} was born in District {districts[i]}."
            chunks[str(3 * i + 2)] = f"District {i} belongs to Region {regions[i]}."
        train = [
            {
                "question": f"In which region was the governor of City {i} born?",
                "support": [str(3 * i), str(3 * people[i] + 1), str(3 * districts[people[i]] + 2)],
                "answer": f"Region {regions[districts[people[i]]]}",
            }
            for i in range(6)
        ]
        target = rng.choice([6, 7])
        yield {
            "id": str(index),
            "family": f"question-{index}",
            "chunks": chunks,
            "train_queries": train,
            "query": {
                "question": f"In which region was the governor of City {target} born?",
                "support": [str(3 * target), str(3 * people[target] + 1), str(3 * districts[people[target]] + 2)],
                "answer": f"Region {regions[districts[people[target]]]}",
            },
            "read_budget": 2,
            "block_bytes": 512,
            "block_slots": 8,
            "storage_budget": 4096,
            "seed": rng.randrange(2**31),
            "search_steps": 10,
        }


def run_case(case, method, answerer=None):
    layout = build_layout(case, method)
    route = router(layout, case["chunks"], case["train_queries"], case["read_budget"])
    indices = (
        list(oracle_cover(layout, case["query"]["support"], case["read_budget"]))
        if method == "gold_routing"
        else route(case["query"]["question"])
    )
    while len(indices) < case["read_budget"]:
        indices.append(len(layout) - 1)
    database = [serialize(block, case["chunks"], case["block_bytes"]) for block in layout]
    public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)
    retrieved = {}
    wire = 0
    for index in indices:
        payload, size = private_read(database, index, public_key, private_key)
        wire += size
        for record in json.loads(payload.rstrip(b"\0").decode()):
            retrieved[record["id"]] = record["text"]
    answer_started = time.perf_counter()
    reply = answerer.answer(case["query"]["question"], retrieved) if answerer else None
    answer_ms = (time.perf_counter() - answer_started) * 1000 if answerer else None
    return {
        "retrieved": sorted(retrieved),
        "answer": None if reply is None else reply["answer"],
        "answer_support": None if reply is None else reply["support_ids"],
        "answerer": None if answerer is None else answerer.config,
        "answer_latency_ms": answer_ms,
        "oracle": list(oracle_cover(layout, case["query"]["support"], case["read_budget"])),
        "layout": [sorted(b) for b in layout],
        "wire_bytes": wire,
        "logical_reads": len(indices),
    }


def score(case, result):
    support = set(case["query"]["support"])
    retrieved = set(result["retrieved"])
    oracle = set().union(*(set(result["layout"][i]) for i in result["oracle"]))
    available = result.get("answer") is not None
    prediction_tokens = tokens(result["answer"]) if available else []
    target_tokens = tokens(case["query"]["answer"])
    predicted, target = Counter(prediction_tokens), Counter(target_tokens)
    overlap = sum((predicted & target).values())
    precision = overlap / sum(predicted.values()) if predicted else 0
    recall = overlap / sum(target.values()) if target else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else int(not predicted and not target)
    special = (["yes"], ["no"], ["noanswer"])
    if (prediction_tokens in special or target_tokens in special) and prediction_tokens != target_tokens:
        f1 = 0.0
    cited = set(result.get("answer_support") or [])
    support_precision = len(cited & support) / len(cited) if cited else 0.0
    support_recall = len(cited & support) / len(support) if support else 1.0
    return {
        "deployed_ec": int(support <= retrieved),
        "oracle_ec": int(support <= oracle),
        "support_recall": len(support & retrieved) / len(support) if support else 1,
        "answer_f1": f1 if available else None,
        "answer_exact_match": int(prediction_tokens == target_tokens) if available else None,
        "answer_support_precision": support_precision if available else None,
        "answer_support_recall": support_recall if available else None,
        "answer_support_f1": 2 * support_precision * support_recall / (support_precision + support_recall) if available and support_precision + support_recall else (0.0 if available else None),
        "answer_latency_ms": result.get("answer_latency_ms"),
        "physical_storage_bytes": case["block_slots"] * case["block_bytes"],
        "online_bytes": result["wire_bytes"],
        "logical_reads": result["logical_reads"],
        "replica_count": sum(map(len, result["layout"])) - len(case["chunks"]),
        "legal_effects": int(support <= retrieved),
    }
