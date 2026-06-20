from __future__ import annotations

import os
import re

from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer


MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
LORA_RANK = int(os.getenv("LORA_RANK", "32"))
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/meta-Llama-3.1-8B-Instruct")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results/unsloth_grpo_baseline")
REWARD_VARIANT = os.getenv("REWARD_VARIANT", "baseline")
MAX_STEPS = int(os.getenv("MAX_STEPS", "250"))


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


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[-1].strip()


def get_gsm8k_questions(split: str = "train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


def _completion_texts(completions) -> list[str]:
    return [completion[0]["content"] for completion in completions]


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = _completion_texts(completions)
    extracted = [extract_xml_answer(response) for response in responses]
    return [2.0 if response == target else 0.0 for response, target in zip(extracted, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = _completion_texts(completions)
    extracted = [extract_xml_answer(response) for response in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = _completion_texts(completions)
    return [0.5 if re.match(pattern, response, flags=re.DOTALL) else 0.0 for response in responses]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = _completion_texts(completions)
    return [0.5 if re.search(pattern, response, flags=re.DOTALL) else 0.0 for response in responses]


def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(response) for response in _completion_texts(completions)]


def length_penalty_reward_func(completions, **kwargs) -> list[float]:
    responses = _completion_texts(completions)
    penalties = []
    for response in responses:
        overflow = max(0, len(response) - 900)
        penalties.append(-min(0.3, overflow * 0.0005))
    return penalties


def reward_functions():
    baseline = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]
    if REWARD_VARIANT == "length_aware":
        return [*baseline, length_penalty_reward_func]
    if REWARD_VARIANT != "baseline":
        raise ValueError(f"Unknown REWARD_VARIANT={REWARD_VARIANT!r}")
    return baseline


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = get_gsm8k_questions()
    max_prompt_length = 256
    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=int(os.getenv("NUM_GENERATIONS", "6")),
        max_prompt_length=max_prompt_length,
        max_completion_length=MAX_SEQ_LENGTH - max_prompt_length,
        max_steps=MAX_STEPS,
        save_steps=MAX_STEPS,
        max_grad_norm=0.1,
        report_to=os.getenv("REPORT_TO", "none"),
        output_dir=OUTPUT_DIR,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions(),
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_lora(os.path.join(OUTPUT_DIR, "grpo_saved_lora"))


if __name__ == "__main__":
    main()
