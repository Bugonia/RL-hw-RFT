# Poster Outline

## Title

Reward-Shaped GRPO for Math Reasoning LLMs

## Panel 1: Motivation

Message:

RFT can improve reasoning, but reward design affects what the model learns.

Visual:

- Base model -> SFT -> GRPO
- highlight that GRPO receives verifiable correctness and format rewards

## Panel 2: Course Baseline

Show:

- Unsloth Llama-3.1-8B GRPO notebook
- GSM8K
- XML answer format
- TRL `GRPOTrainer`

Small pipeline:

```text
GSM8K prompt -> 6 sampled completions -> reward functions -> GRPO update
```

## Panel 3: Our Variant

Show formula:

```text
R_variant = R_baseline + length_penalty
```

Explain in one sentence:

The penalty discourages excessive completion length while keeping correctness
as the main reward.

## Panel 4: Experimental Setup

Compact table:

- model
- dataset
- LoRA rank
- training steps
- methods: Base, SFT, GRPO baseline, GRPO length-aware
- evaluation metric: exact GSM8K answer accuracy

## Panel 5: Results

Main chart:

- GSM8K accuracy by method

Secondary chart:

- format compliance and average response length

Optional chart:

- reward components during training

## Panel 6: Examples

Use 2 short examples:

- one success case
- one failure or reward-hacking case

Keep text very short. Show only final answer extraction and diagnosis.

## Panel 7: Takeaways

Three bullets:

- GRPO gives a direct way to optimize verifiable reasoning rewards.
- SFT and GRPO should be compared under the same prompt/data/LoRA setting.
- Small reward changes can affect formatting, verbosity, and answer accuracy.

## Visual Style

Use a clean technical poster:

- white or near-white background
- one accent color for GRPO
- one contrasting color for the reward variant
- compact plots and tables
- no dense paragraphs

