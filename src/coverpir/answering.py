"""Trusted client-local multi-hop answerers over recovered PIR records."""

import json
from pathlib import Path
import subprocess


def request(question, retrieved):
    return {
        "protocol": "coverpir.answer.v1", "question": question,
        "evidence": [{"id": str(key), "text": text} for key, text in sorted(retrieved.items())],
    }


def validate_answer(output, retrieved):
    if not isinstance(output, dict) or not isinstance(output.get("answer"), str):
        raise ValueError("Answerer must return a JSON object with an answer string")
    support = output.get("support_ids", [])
    if not isinstance(support, list) or any(not isinstance(key, str) for key in support):
        raise ValueError("support_ids must be a list of recovered chunk IDs")
    if not set(support) <= set(retrieved):
        raise ValueError("Answerer cited evidence outside the recovered records")
    return {"answer": output["answer"], "support_ids": sorted(set(support))}


class CommandAnswerer:
    def __init__(self, config):
        self.config = dict(config)
        command = config.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError("Answerer command must be a nonempty local argv list")

    def answer(self, question, retrieved):
        response = subprocess.run(
            self.config["command"], input=json.dumps(request(question, retrieved)),
            text=True, encoding="utf-8", capture_output=True, check=True,
            timeout=float(self.config.get("timeout_seconds", 120)),
            cwd=self.config.get("cwd"), shell=False,
        )
        return validate_answer(json.loads(response.stdout), retrieved)


class TransformersAnswerer:
    """Load an existing causal instruction model; never fetch weights/tokenizers."""

    def __init__(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = dict(config)
        path = Path(config["model_path"]).resolve()
        if not path.is_dir():
            raise FileNotFoundError("Expected an existing local model directory: " + str(path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True
        ).to(config.get("device", "cpu")).eval()
        self.torch = torch

    def answer(self, question, retrieved):
        message = (
            "Answer the question by combining the supplied evidence across all required hops. "
            "Use only these records. Return exactly one JSON object with keys answer "
            "(a short answer string) and support_ids (the supporting record IDs). "
            "If the evidence is insufficient, return an empty answer and empty support_ids.\n"
            + json.dumps(request(question, retrieved), ensure_ascii=False)
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        maximum = int(self.config.get("max_new_tokens", 128))
        model_context = getattr(self.model.config, "max_position_embeddings", None) or 4096
        context = min(int(self.config.get("context_tokens", model_context)), int(model_context))
        if inputs["input_ids"].shape[-1] + maximum > context:
            raise ValueError("Recovered evidence exceeds the registered answerer context; use smaller blocks")
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=maximum, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        return validate_answer(json.loads(text), retrieved)


def load_answerer(path):
    if path is None:
        return None
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "cwd" in config:
        config["cwd"] = str((config_path.parent / config["cwd"]).resolve())
    backend = config.get("backend", "command")
    if backend == "command":
        return CommandAnswerer(config)
    if backend == "transformers":
        config["model_path"] = str((config_path.parent / config["model_path"]).resolve())
        return TransformersAnswerer(config)
    raise ValueError("Answerer backend must be command or transformers")
