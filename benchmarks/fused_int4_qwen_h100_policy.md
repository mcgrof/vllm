# Qwen2.5-7B-Instruct fused INT4 policy on H100

Use `VLLM_FUSED_INT4_MIN_SEQ_LEN=48` as the current default policy for the
Qwen2.5-7B-Instruct H100 lane.

## Why 48
The H100 policy sweep showed:
- `MSL=40` still failed at longer prompts
- `MSL=48` achieved perfect text match on the tested sweep set
- `MSL=64` was safe but unnecessarily conservative

## What value 48 recovers versus 64
Using 48 instead of 64:
- reduces the FP16 protected / shadow window by **16 decode positions per sequence**
- shrinks that protected window by **25%** relative to 64
- enters the fused INT4 path earlier while preserving the tested Qwen/H100 correctness envelope

## What this does not mean
This is not a universal fused-INT4 default for all models.
It is a model-specific policy outcome derived from the Qwen/H100 sweep.
Other models should be calibrated separately.

## Next scaling step after the A100/W7900 reconciliation
The next practical step is to keep scaling on the NVIDIA lane, not to spend more
time on ROCm/W7900 confusion. Concretely:
- keep Qwen2.5/H100 `MSL=48` as the current tested policy,
- run the same strict policy methodology on Qwen2-7B and Mistral-7B,
- then move from single-GPU policy calibration into bounded decode-throughput
  checks and API-server / lm-eval validation.

The important testing discipline is to keep correctness strict: exact text/token
agreement or an explicitly documented semantic rule. Prefix-only correctness is
not enough.
