# Mini Paper Outline

Recommended title:

Reward-Shaped GRPO for Mathematical Reasoning in Large Language Models

## Abstract

One paragraph:

- Reinforcement fine-tuning can improve LLM reasoning when rewards are
  verifiable.
- We reproduce the course-recommended Unsloth GRPO baseline on GSM8K.
- We compare the base model, SFT, GRPO, and a length-aware GRPO reward variant.
- We report answer accuracy, format compliance, response length, reward
  components, and qualitative failure cases.

## 1. Introduction

Key points:

- The course topic asks for RL fine-tuning of LLMs with an RL-level novelty.
- GRPO is attractive because it avoids a separate value model and uses grouped
  relative rewards.
- For math reasoning, exact-answer verification gives a clean reward signal.
- Reward design still matters: formatting rewards can dominate, and long
  completions can hide answer errors.

End with contributions:

- We reproduce an official Unsloth GRPO baseline for GSM8K.
- We implement a matched SFT baseline.
- We test a length-aware reward variant under the same GRPO setting.
- We analyze accuracy, format compliance, response length, and examples.

## 2. Background

### 2.1 RFT and GRPO

Briefly explain:

- model samples multiple completions for each prompt
- reward functions score each completion
- group-relative advantages compare completions from the same prompt
- policy is updated toward high-reward completions

### 2.2 Course Baselines

Describe the two provided resources:

- Unsloth Llama-3.1-8B GRPO notebook
- Berkeley LLM RL project with DPO, IPO, AOT, reward modeling, GRPO, DrGRPO,
  and GSPO

## 3. Method

### 3.1 Baseline Reproduction

State the exact baseline:

- `unsloth/meta-Llama-3.1-8B-Instruct`
- GSM8K
- 4-bit LoRA, rank 32
- XML reasoning/answer format
- TRL `GRPOTrainer`
- reward stack: XML count, soft format, strict format, integer answer,
  correctness

### 3.2 SFT Baseline

Use the same prompt and output format. Train on GSM8K reasoning and final
answers transformed into XML.

### 3.3 Length-Aware GRPO Variant

Define:

```text
R_variant = R_baseline - min(lambda * overflow_chars, penalty_cap)
```

Explain:

- correctness remains the dominant reward
- the penalty only discourages unnecessarily long completions
- all other training settings remain fixed

## 4. Experiments

### 4.1 Setup

Report:

- GPU/Colab environment
- training steps
- LoRA configuration
- max sequence length
- number of generations per prompt
- evaluation subset size

### 4.2 Main Results

Table:

- Base
- SFT
- GRPO baseline
- GRPO length-aware reward

Metrics:

- GSM8K accuracy
- format compliance
- average response length
- invalid answer rate

### 4.3 Reward Analysis

Plot or table:

- correctness reward
- XML count reward
- format rewards
- length penalty
- total reward

Question to answer:

- did the variant change the reward distribution in the intended direction?

### 4.4 Qualitative Examples

Include 2-4 examples:

- GRPO fixes a base-model mistake
- SFT gets format right but answer wrong
- baseline GRPO is verbose or malformed
- length-aware GRPO improves or fails

## 5. Discussion

Discuss:

- whether GRPO beat SFT
- whether the reward variant helped accuracy or mainly improved formatting
- possible reward hacking
- compute limitations
- why DrGRPO/GSPO from the Berkeley reference would be natural next steps

## 6. Conclusion

Restate:

- the course-recommended GRPO baseline was reproduced
- the comparison with SFT was controlled
- the reward variant gave evidence about reward-design sensitivity

## Appendix

Suggested appendix:

- exact commands
- hyperparameters
- reward-function code
- additional samples
- failed runs and debugging notes

