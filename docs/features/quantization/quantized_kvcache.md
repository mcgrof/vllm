# Quantized KV Cache

## FP8 KV Cache Overview

Efficient memory usage is crucial for working with large language models. Quantizing the KV (Key-Value) cache to FP8 format can significantly reduce its memory footprint. This optimization enables you to store more tokens in memory, leading to improved throughput and support for longer context windows.

> **Note:** When using the Flash Attention 3 backend with FP8 KV cache, attention operations are also performed in the quantized (FP8) domain. In this configuration, queries are quantized to FP8 in addition to keys and values.

### Supported FP8 KV-Cache Quantization Schemes

vLLM supports two main quantization strategies for the FP8 KV-cache:

- **Per-tensor quantization:**  
  A single scale is applied for each Q, K, and V tensor individually. (`q/k/v_scale = [1]`)
- **Per-attention-head quantization:**  
  Each scale corresponds to an attention head: `q_scale = [num_heads]`, `k/v_scale = [num_kv_heads]`.

> **Note:**  
> Per-attention-head quantization is currently available **only with the Flash Attention backend** and requires the calibration pathway provided by **llm-compressor**.

### Scale Calibration Approaches

You can configure how the quantization scales are computed in vLLM using three different approaches:

1. **No calibration (default scales):**  
   All quantization scales are set to `1.0`.  
   _Configure with:_  
   ```python
   kv_cache_dtype="fp8"
   calculate_kv_scales=False
   ```

2. **Random token calibration (on-the-fly):**  
   Scales are automatically estimated from a single batch of random tokens during warmup and then fixed.  
   _Configure with:_  
   ```python
   kv_cache_dtype="fp8"
   calculate_kv_scales=True
   ```

3. **[Recommended] Calibration with a dataset (via llm-compressor):**  
   Scales are estimated using a curated calibration dataset for maximum accuracy.  
   This requires the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.  
   _See example below!_

#### Additional `kv_cache_dtype` Options

- `kv_cache_dtype="auto"`: Use the model's default data type
- `kv_cache_dtype="fp8_e4m3"`: Supported on CUDA 11.8+ and ROCm (AMD GPUs)
- `kv_cache_dtype="fp8_e5m2"`: Supported on CUDA 11.8+

---

## Examples

### 1. No Calibration (`kv_cache_dtype="fp8"`, `calculate_kv_scales=False`)

All quantization scales are set to 1.0.

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=False,
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 2. Random Token Calibration (`kv_cache_dtype="fp8"`, `calculate_kv_scales=True`)

Scales are automatically estimated from a single batch of tokens during warmup.

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True,
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 3. **[Recommended] Calibration Using a Dataset (with `llm-compressor`)**

For the highest-quality quantization, we recommend calibrating against a dataset using `llm-compressor`. This enables advanced strategies such as per-attention-head quantization.

#### Install the required package

```bash
pip install llmcompressor
```

#### Example: Quantize Llama Attention & KV Cache to FP8

```python
"""
Quantize Llama attention + KV cache to FP8 (choose either 'tensor' or 'attn_head' strategy)
using llm-compressor one-shot calibration.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
STRATEGY = "tensor"       # or "attn_head"
NUM_CALIB_SAMPLES = 512   # Good starting value
MAX_SEQ_LEN = 2048

# -----------------------------
# Helpers
# -----------------------------
def process_and_tokenize(example, tokenizer: AutoTokenizer):
    """Convert chat messages to tokens."""
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    )

def build_recipe(strategy: str) -> QuantizationModifier:
    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=strategy)
    return QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["LlamaAttention"],  # Quantize queries: q_scale
                input_activations=fp8_args,
            )
        },
        kv_cache_scheme=fp8_args,           # Quantize KV cache: k/v_scale
    )

# -----------------------------
# Main
# -----------------------------
def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIB_SAMPLES}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(
        lambda ex: process_and_tokenize(ex, tokenizer),
        remove_columns=ds.column_names,
    )

    recipe = build_recipe(STRATEGY)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIB_SAMPLES,
    )

    save_dir = f"{MODEL_ID.rstrip('/').split('/')[-1]}-kvattn-fp8-{STRATEGY}"
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
```

For more detailed and up-to-date examples, see the [`llm-compressor` official examples](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache).

---

## INT4 Fused KV Cache (Experimental)

The INT4 fused KV cache stores keys and values as packed 4-bit integers (2 values
per uint8 byte) with per-group FP16 scales (group_size=32). During decode, a
fused Triton kernel dequantizes the INT4 data in-register and computes attention
without materializing an FP16 intermediate in global memory. This eliminates the
extra memory traffic that makes non-fused quantization counterproductive.

The fused path reduces KV cache memory footprint by approximately 4x compared to
FP16 and delivers decode speedups of 2.5x-5.4x at batch sizes >= 2. At batch
size 1, the overhead of the Triton kernel can exceed the traffic savings, so a
bounded dispatch policy (falling back to FlashAttention at B=1) is planned.

The research behind this feature is published in
[Memory-Traffic Saturation in Autoregressive Transformer Decode]((anonymized — see paper supplement)),
which benchmarks 14 open-weight models across W7900, A100, H100, and B200
and shows that kernel fusion — not quantization alone — is the mechanism that
turns compression into real decode speedup.

### Usage

```bash
# CLI
vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-cache-dtype int4_fused

# Python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="int4_fused",
)
out = llm.generate("London is the capital of", SamplingParams(temperature=0.7))
print(out[0].outputs[0].text)
```

### How it works

1. **Cache write (prefill)**: FP16/BF16 K/V outputs from the model are quantized
   to INT4 by the `reshape_and_cache_int4` Triton kernel. Each group of 32
   elements is independently scaled: `scale = max(|group|) / 7`. Two INT4
   values are packed into one uint8 byte (low nibble = even index, high nibble =
   odd index). Scales are stored as FP16 in a separate tensor.

2. **Cache read (decode)**: The `fused_int4_decode` Triton kernel reads packed
   uint8 bytes from the paged KV cache, unpacks the nibbles, multiplies by the
   per-group scale, and computes the QK dot product and softmax-weighted V
   accumulation — all in-register. No FP16 buffer is written to global memory.

3. **Prefill fallback**: During prefill (multi-token queries), the fresh K/V
   are still in FP16 from the model. The backend uses `torch.nn.functional.scaled_dot_product_attention`
   (SDPA) directly on the FP16 data and logs a warning that prefill is not using
   the fused path. Only decode uses the fused INT4 kernel.

### Asymmetric K/V support (future)

The implementation stores K and V scales in separate tensors (`k_scales`,
`v_scales`). The decode kernel has independent scale loading paths for K and V.
This structure supports future asymmetric quantization where keys and values
use different precision — motivated by the finding that values universally
tolerate INT4 while some model families (Qwen) require higher key precision.

### Backend verification

When `int4_fused` is selected, vLLM logs machine-readable verification lines:

```
Backend manifest: requested_backend=int4_fused, selected_backend=FUSED_INT4, ...
[FusedInt4] Backend verification: selected_backend=FUSED_INT4, decode_kernel=fused_int4_triton, ...
```

These logs allow benchmark harnesses to confirm the fused path actually ran
rather than falling back silently.

### Constraints

- **Head size** must be divisible by 32 (the group size)
- **Sliding window attention** is not yet supported
- **ALiBi** positional encoding is not yet supported
- **CUDAGraph** capture is not yet supported (decode runs without graph capture)
- Prefill uses SDPA fallback, not the fused kernel

### Validated hardware

| GPU | Platform | Triton | Status |
|-----|----------|--------|--------|
| AMD Radeon Pro W7900 | ROCm 6.4 | 3.5.1 | Tested: 2.5x-5.4x decode speedup, cos_sim=1.0 |
| NVIDIA H100 | CUDA | — | Planned |

### Smoke benchmark

A self-contained kernel-level benchmark is included:

```bash
python benchmarks/fused_int4_smoke.py
```

This exercises both the FP16 SDPA baseline and the fused INT4 decode kernel
with synthetic tensors (no model download needed). It emits a JSON manifest
to stdout with per-point latencies, speedup ratios, cosine similarity, and
backend verification fields.

### Additional `kv_cache_dtype` option

- `kv_cache_dtype="int4_fused"`: Packed INT4 with fused in-kernel dequantization (experimental)
