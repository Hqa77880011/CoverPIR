"""Finite joining/generator/scoring checks; no PIR or language-model execution."""

import ast
from collections import Counter
from pathlib import Path
import random
import re
import string
import unittest

import numpy as np

from coverpir.answering import FiniteRelationsAnswerer, load_answerer


ROOT = Path(__file__).resolve().parents[1]


def core_functions():
    source = (ROOT / "src/coverpir/core.py").read_text(encoding="utf-8")
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
             and node.name in ("generate", "tokens", "score")]
    namespace = dict(np=np, Counter=Counter, re=re, string=string)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "core-pure-functions", "exec"), namespace)
    return namespace


class RelationTests(unittest.TestCase):
    def setUp(self):
        self.reader = FiniteRelationsAnswerer({"backend": "finite_relations"})
        self.question = "In which region was the governor of City 6 born?"
        self.evidence = {"city": "City 6 is governed by Person 2.",
                         "person": "Person 2 was born in District 5.",
                         "district": "District 5 belongs to Region 1.",
                         "distractor": "District 6 belongs to Region 6."}

    def test_actual_three_hop_join_does_not_guess_city_number(self):
        self.assertEqual(self.reader.answer(self.question, self.evidence),
                         dict(answer="Region 1", support_ids=["city", "district", "person"]))

    def test_any_missing_hop_returns_no_answer_or_fabricated_support(self):
        for missing in ("city", "person", "district"):
            incomplete = {key: value for key, value in self.evidence.items() if key != missing}
            self.assertEqual(self.reader.answer(self.question, incomplete), dict(answer="", support_ids=[]))

    def test_ambiguous_relation_returns_empty(self):
        conflict = dict(self.evidence, conflict="Person 2 was born in District 3.")
        self.assertEqual(self.reader.answer(self.question, conflict), dict(answer="", support_ids=[]))

    def test_registered_backend_and_permuted_generated_truth(self):
        reader = load_answerer(ROOT / "configs/finite_reader.json")
        functions = core_functions()
        for case in functions["generate"](random.Random(7), 3):
            for query in case["train_queries"] + [case["query"]]:
                result = reader.answer(query["question"], case["chunks"])
                self.assertEqual(result["answer"], query["answer"])
                self.assertEqual(set(result["support_ids"]), set(query["support"]))

    def test_real_answer_and_support_enter_exact_match_f1(self):
        functions = core_functions()
        case = next(functions["generate"](random.Random(7), 1))
        reply = self.reader.answer(case["query"]["question"], case["chunks"])
        result = dict(retrieved=list(case["chunks"]), answer=reply["answer"],
                      answer_support=reply["support_ids"], layout=[list(case["chunks"])],
                      oracle=[0], wire_bytes=0, logical_reads=2)
        metrics = functions["score"](case, result)
        self.assertEqual(metrics["answer_exact_match"], 1)
        self.assertEqual(metrics["answer_f1"], 1)
        self.assertEqual(metrics["answer_support_f1"], 1)
        result.update(answer="", answer_support=[])
        self.assertEqual(functions["score"](case, result)["answer_exact_match"], 0)


if __name__ == "__main__":
    unittest.main()
