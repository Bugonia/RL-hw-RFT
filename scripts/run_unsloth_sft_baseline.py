from __future__ import annotations

import os

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
LORA_RANK = int(os.getenv("LORA_RANK", "32"))
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/meta-Llama-3.1-8B-Instruct")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results/unsloth_sft_baseline")
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


def extract_hash_answer(text: str) -> str:
    if "####" not in text:
        return text.strip()
    return text.split("####")[-1].strip()


def format_example(example: dict[str, str]) -> dict[str, str]:
    answer = extract_hash_answer(example["answer"])
    reasoning = example["answer"].split("####")[0].strip()
    text = (
        f"{SYSTEM_PROMPT.strip()}\n\n"
        f"Question:\n{example['question']}\n\n"
        "<reasoning>\n"
        f"{reasoning}\n"
        "</reasoning>\n"
        "<answer>\n"
        f"{answer}\n"
        "</answer>\n"
    )
    return {"text": text}


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
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

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_steps=MAX_STEPS,
        optim="paged_adamw_8bit",
        report_to=os.getenv("REPORT_TO", "none"),
    )

    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
    trainer.train()
    model.save_pretrained(os.path.join(OUTPUT_DIR, "sft_lora"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "sft_lora"))


if __name__ == "__main__":
    main()
