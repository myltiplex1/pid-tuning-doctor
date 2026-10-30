"""Turn the PID Tuning Doctor dataset into the Gemma chat format used in the
course (lab 5.6): a dict {"prompts": [...], "responses": [...]} with
<start_of_turn>/<end_of_turn> delimiters.

    python prepare_gemma_data.py            # reads data/, writes data/gemma_*.json
"""

import json
import os

INSTRUCTION = ("You are a PID tuning assistant for control students. "
               "Read the step-response description and reply with the diagnosis, "
               "the evidence you used, and one fix.")

SOT, EOT = "<start_of_turn>", "<end_of_turn>"


def format_prompt(text: str) -> str:
    """Prompt only (use this at evaluation time)."""
    return f"{SOT}user\n{INSTRUCTION}\n\n{text}{EOT}\n"


def format_response(text: str) -> str:
    return f"{SOT}model\n{text}{EOT}"


def convert(split, data_dir="data"):
    rows = json.load(open(os.path.join(data_dir, f"{split}.json")))
    out = {
        "prompts": [format_prompt(r["input"]) for r in rows],
        "responses": [format_response(r["output"]) for r in rows],
        "labels": [r["label"] for r in rows],
        "plants": [r["plant"] for r in rows],
    }
    path = os.path.join(data_dir, f"gemma_{split}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{split:7s} {len(rows):4d} examples -> {path}")


if __name__ == "__main__":
    for s in ["train", "val", "test_A", "test_B"]:
        convert(s)
