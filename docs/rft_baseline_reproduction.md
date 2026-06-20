# RFT Baseline Reproduction Plan

This document records how the project uses the two course-provided baseline
resources:

1. Unsloth Llama-3.1-8B GRPO Colab notebook
2. Berkeley Deep RL LLM RL final project PDF/codebase

The main course-project path is the Unsloth GRPO baseline because the course
slides explicitly mention that the Colab can be opened directly from campus and
is suitable for quick reproduction.

## 1. Unsloth GRPO Baseline

### Source

Course slide link:

```text
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb
```

Raw notebook used for implementation reference:

```text
https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Llama3.1_(8B)-GRPO.ipynb
```

### Baseline Components

The notebook uses:

- `unsloth.FastLanguageModel`
- `unsloth/meta-Llama-3.1-8B-Instruct`
- `load_in_4bit = True`
- LoRA rank 32
- `max_seq_length = 1024`
- GSM8K from `openai/gsm8k`
- XML response format with `<reasoning>` and `<answer>` tags
- TRL `GRPOTrainer`

The reward stack contains:

- XML count reward
- soft format reward
- strict format reward
- integer-answer reward
- exact correctness reward

The default training configuration contains:

- `learning_rate = 5e-6`
- `weight_decay = 0.001`
- `warmup_ratio = 0.1`
- `lr_scheduler_type = "cosine"`
- `optim = "paged_adamw_8bit"`
- `num_generations = 6`
- `max_prompt_length = 256`
- `max_steps = 250`
- `max_grad_norm = 0.1`

### Local Script

The project version is:

```bash
python3 scripts/run_unsloth_grpo_baseline.py
```

This script is meant to run in Colab or on the provided 4090 machine after
installing the Unsloth/TRL stack. It will not run in this local CPU-only
workspace unless those GPU dependencies are installed.

### Required Reproduction Outputs

Save:

- training logs
- final LoRA adapter
- pre-training accuracy
- post-training accuracy
- reward-component curves
- several qualitative examples

Suggested paths:

```text
results/unsloth_grpo_baseline/
results/unsloth_grpo_length_reward/
```

## 2. SFT Baseline

The course slide asks us to compare RFT with SFT under the same setting.

Use the same:

- model family
- GSM8K split
- prompt format
- LoRA setup
- maximum sequence length

The target output should map GSM8K answers into:

```text
<reasoning>
GSM8K rationale text
</reasoning>
<answer>
final numeric answer
</answer>
```

Report SFT accuracy with the same XML answer extractor used for GRPO.

Project script:

```bash
python3 scripts/run_unsloth_sft_baseline.py
```

Suggested quick pilot:

```bash
MAX_STEPS=50 OUTPUT_DIR=results/unsloth_sft_pilot \
python3 scripts/run_unsloth_sft_baseline.py
```

## 3. Proposed RL-Level Variation

The simplest controlled novelty is a length-aware reward variant:

```text
reward = baseline_rewards + length_penalty
```

The length penalty should be small. It is not meant to dominate answer
correctness. It only discourages completions that are unnecessarily long or that
place excessive text after the final answer.

Main comparison:

- GRPO baseline reward stack
- GRPO length-aware reward stack

Keep all other settings fixed.

Project command:

```bash
REWARD_VARIANT=length_aware OUTPUT_DIR=results/unsloth_grpo_length_reward \
python3 scripts/run_unsloth_grpo_baseline.py
```

## 4. Evaluation

Use one exact-answer evaluator for all methods:

```bash
python3 scripts/evaluate_gsm8k_xml.py \
  --model_name unsloth/meta-Llama-3.1-8B-Instruct \
  --limit 100 \
  --output_jsonl results/base_gsm8k_eval.jsonl
```

For LoRA checkpoints, merge or load the adapter according to the Unsloth
notebook's inference section, then run the same evaluation prompt and answer
extraction logic.

## 5. Berkeley Baseline Reference

Course slide link:

```text
https://rail.eecs.berkeley.edu/deeprlcourse/static/misc/llm_rl_default_final_project.pdf
```

The Berkeley project gives a more complete RLHF baseline with:

- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- preference dataset: WildChat judged 5k
- offline methods: DPO, IPO, AOT
- reward model: Bradley-Terry pairwise reward model
- online methods: GRPO, DrGRPO, GSPO
- key files for implementation: logprobs, rollout buffer, offline losses,
  reward-model training, and GRPO-family updates

Use it as the theoretical and experimental reference for:

- group-relative advantage normalization
- sampled KL regularization
- token-level GRPO clipping
- DrGRPO reward-centering behavior
- GSPO sequence-level clipping

## 6. Report Framing

The mini paper should not claim that we invented GRPO. It should say:

1. We reproduced the course-recommended GRPO baseline.
2. We compared GRPO with SFT under a matched setting.
3. We tested a small reward-design variant.
4. We analyzed when the variant helped or failed.
