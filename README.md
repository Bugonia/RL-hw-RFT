# Reinforcement Fine-Tuning LLMs Course Project

This repository is a course project skeleton for Topic 4: reinforcement
learning in large language models.

The primary direction is Reinforcement Fine-Tuning (RFT) for LLM reasoning,
following the baseline resources listed in the course slides:

- Unsloth Llama-3.1-8B GRPO notebook
- Berkeley Deep RL LLM RL final project baseline

The project first reproduces a GRPO baseline on GSM8K, compares it with an SFT
baseline under a matched setting, and then tests a small RL-level modification
to the reward design. We are only choosing the first subdirection under Topic
4, not the environment-scaling subdirection.

## Core Question

Can GRPO-style reinforcement fine-tuning improve math reasoning accuracy over
SFT under the same model, data, and LoRA setting, and can reward-shaping changes
make the RL signal more stable?

## Planned Contributions

1. Reproduce the Unsloth GRPO baseline on GSM8K.
2. Train an SFT baseline with the same model family and dataset.
3. Compare base, SFT, and GRPO accuracy before and after training.
4. Test one RL-level variation, such as length-aware correctness reward or
   reward-component ablation.
5. Analyze reward components, output format compliance, and qualitative failure
   cases.

## Directory Layout

```text
.
├── docs/
│   ├── experiment_plan.md
│   ├── mini_paper_outline.md
│   ├── rft_baseline_reproduction.md
│   └── poster_outline.md
├── scripts/
│   ├── evaluate_gsm8k_xml.py
│   ├── run_unsloth_sft_baseline.py
│   ├── run_unsloth_grpo_baseline.py
│   └── run_reward_smoke.py
├── src/
│   └── rft_project/
│       └── gsm8k_rewards.py
├── proposal.md
└── requirements.txt
```

## Smoke Test

Reward parsing uses only the Python standard library.

```bash
python3 scripts/run_reward_smoke.py
```

## SFT Baseline

```bash
python3 scripts/run_unsloth_sft_baseline.py
```

## GRPO Baseline

```bash
python3 scripts/run_unsloth_grpo_baseline.py
```

The SFT and GRPO scripts require a GPU environment with the Unsloth/TRL stack
installed. They are intended for Colab or the provided 4090 machine, not for
this local CPU-only project directory.

## Evaluation

```bash
python3 scripts/evaluate_gsm8k_xml.py \
  --model_name unsloth/meta-Llama-3.1-8B-Instruct \
  --limit 100 \
  --output_jsonl results/base_gsm8k_eval.jsonl
```

Use the same evaluator for the base model, SFT checkpoint, GRPO checkpoint, and
reward-variant checkpoint.

See `docs/rft_baseline_reproduction.md` for the exact reproduction plan.
