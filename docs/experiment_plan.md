# Experiment Plan

## Goal

Reproduce the course-recommended GRPO baseline for LLM math reasoning, compare
it with SFT, and test one controlled RL reward-design variation.

## Model Choices

Primary baseline:

- `unsloth/meta-Llama-3.1-8B-Instruct`
- 4-bit loading
- LoRA rank 32
- maximum sequence length 1024

Fallback if 8B is too slow:

- Qwen2.5-1.5B-Instruct, following the Berkeley baseline design

## Training Conditions

### Condition A: No Training

Evaluate the base model directly on GSM8K test examples.

Purpose:

- establish starting accuracy
- establish answer-format and response-length behavior

### Condition B: SFT

Train on GSM8K using the same XML reasoning/answer format as the GRPO prompt.

Purpose:

- compare RL against supervised training under a matched task setting
- satisfy the course requirement to compare SFT and RFT/RL

### Condition C: GRPO Baseline

Reproduce the Unsloth GRPO notebook.

Purpose:

- satisfy the required RFT baseline reproduction
- measure training-before/after accuracy

### Condition D: GRPO Reward Variant

Run GRPO with the same model, data, and budget, but add length-aware reward
shaping.

Purpose:

- test whether reward design changes correctness, format compliance, and
  response length
- provide the required RL-level novelty

## Dataset

Primary dataset:

- GSM8K main split

Prompt format:

```text
<reasoning>
...
</reasoning>
<answer>
...
</answer>
```

Evaluation extracts the final value from the XML `<answer>` block.

## Baseline Hyperparameters

From the Unsloth notebook:

- `learning_rate = 5e-6`
- `adam_beta1 = 0.9`
- `adam_beta2 = 0.99`
- `weight_decay = 0.001`
- `warmup_ratio = 0.1`
- `lr_scheduler_type = cosine`
- `optim = paged_adamw_8bit`
- `num_generations = 6`
- `max_prompt_length = 256`
- `max_steps = 250`
- `max_grad_norm = 0.1`

## Berkeley Reference Settings

If we use the Berkeley codebase instead of the Unsloth Colab path, keep its
benchmark fixed:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- max prompt tokens: 700
- max new tokens for online rollouts: 256
- max response tokens for scoring: 512
- online algorithms: GRPO, DrGRPO, GSPO
- compare against DPO, IPO, AOT and reward-model baselines

## Main Tables

### Table 1: Accuracy

Columns:

- Method
- GSM8K train subset accuracy
- GSM8K test subset accuracy
- Format compliance
- Average response length

Rows:

- Base
- SFT
- GRPO baseline
- GRPO length-aware reward

### Table 2: Reward Components

Columns:

- Method
- Correctness reward
- Integer reward
- Soft format reward
- Strict format reward
- XML count reward
- Length penalty

### Table 3: Qualitative Failure Modes

Columns:

- Error type
- Example prompt
- Baseline output
- Variant output
- Diagnosis

## Expected Claims

A strong result:

- GRPO improves over SFT and base model on GSM8K
- length-aware reward keeps correctness similar while reducing malformed or
  excessively long responses
- reward-component logs explain why the modification helps

A still-acceptable result:

- GRPO reproduces successfully, but the reward variant does not improve final
  accuracy
- the report gives a clear analysis of why the variant helped or failed

## Risk Control

If full GRPO training is too slow:

1. Run the official Unsloth Colab for fewer steps.
2. Evaluate base, SFT, and partial GRPO checkpoints on the same subset.
3. Report the compute limitation honestly.
4. Use reward-function ablations and qualitative samples to support the
   analysis.

