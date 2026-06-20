# Project Proposal: Reward-Shaped GRPO for Math Reasoning LLMs

## 1. Motivation

The course topic asks us to use reinforcement learning to fine-tune LLMs and to
compare RFT with SFT under the same setting. The recommended baseline is the
Unsloth GRPO notebook for Llama-3.1-8B on GSM8K, with the Berkeley Deep RL LLM
RL project as a more detailed reference for GRPO, DrGRPO, GSPO, reward models,
and evaluation design.

This project focuses on a practical question: after reproducing the GRPO
baseline, can a small change to the reward design improve reasoning accuracy or
training stability compared with the original reward stack?

## 2. Research Question

Does GRPO improve GSM8K math reasoning accuracy over SFT, and does a
length-aware reward variant reduce formatting or verbosity failures without
hurting answer correctness?

## 3. Proposed Method

We use the Unsloth GRPO pipeline as the required baseline:

- model: Llama-3.1-8B-Instruct through Unsloth
- dataset: GSM8K
- training: 4-bit LoRA with TRL `GRPOTrainer`
- response format: XML reasoning and answer tags
- rewards: XML count, soft format, strict format, integer answer, correctness

We then add one controlled RL-level variation:

- baseline reward stack: the original Unsloth reward functions
- proposed reward stack: correctness and format rewards plus a small length
  penalty for unnecessarily long or malformed completions

This keeps the model, data, LoRA configuration, and training budget fixed while
changing the reward signal.

## 4. Baselines

We compare:

1. Base model before additional training
2. SFT on GSM8K under the same prompt/answer format
3. GRPO baseline reproduced from the Unsloth notebook
4. GRPO with length-aware reward shaping

The Berkeley baseline is used as a method reference for GRPO-family analysis,
especially group-relative advantage normalization, KL regularization, DrGRPO,
and GSPO.

## 5. Experimental Design

Main evaluation:

- GSM8K test accuracy
- exact final-answer accuracy after XML answer extraction
- format compliance rate
- average response length
- reward-component means during GRPO training

Optional diagnostics:

- compare short and long reasoning failures
- inspect samples where format reward is high but correctness is wrong
- inspect samples where correctness is right but formatting is broken

## 6. Expected Novelty

The novelty is deliberately small but RL-specific:

- we do not change the base model or dataset
- we change the reward design used by GRPO
- we evaluate whether the reward modification changes accuracy, format
  compliance, and response length
- we compare against SFT and the reproduced GRPO baseline

This matches the course requirement for RL-level novelty while staying feasible
under the deadline.

## 7. Deliverables

- Code package with GRPO reproduction script, SFT baseline script, GSM8K
  evaluator, and reward functions
- Mini paper using the NeurIPS template
- Poster in PPT or PDF format
- Experiment logs, tables, and ablation results
