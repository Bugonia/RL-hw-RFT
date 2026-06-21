from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str:
    if "####" not in text:
        return text.strip()
    return text.split("####")[-1].strip()


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().replace(",", "").split())


def build_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT.strip()}\n\nQuestion:\n{question}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="unsloth/meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_jsonl", default="results/gsm8k_eval.jsonl")
    args = parser.parse_args()

    tokenizer_source = args.adapter_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in dataset:
            prompt = build_prompt(example["question"])
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            pred = normalize_answer(extract_xml_answer(completion))
            gold = normalize_answer(extract_hash_answer(example["answer"]))
            is_correct = pred == gold
            correct += int(is_correct)
            total += 1
            handle.write(
                json.dumps(
                    {
                        "question": example["question"],
                        "gold": gold,
                        "prediction": pred,
                        "correct": is_correct,
                        "completion": completion,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print({"correct": correct, "total": total, "accuracy": correct / max(total, 1)})
    print(f"Wrote detailed outputs to {output_path}")


if __name__ == "__main__":
    main()
