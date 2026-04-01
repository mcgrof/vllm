#!/usr/bin/env python3
"""A0 generation-drift harness for fused INT4 KV cache.

Compares FP16 vs fused INT4 generation on Qwen/Qwen2.5-1.5B to
characterize the generation-quality catastrophe observed on H100
(GSM8K 0.6088 -> 0.0023).

The harness produces:
  - first divergent token index per prompt
  - top-k logprob ranking at divergence point
  - per-head K-side vs V-side quant-dequant error
  - whether divergence is immediate vs compounds over decode steps
  - overall summary statistics

Output: JSON to stdout.

Usage:
    python benchmarks/fused_int4_drift_harness.py [--max-tokens 200] [--model Qwen/Qwen2.5-1.5B]
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Prompt battery
# ---------------------------------------------------------------------------

# 50 GSM8K-style problems (abbreviated but representative)
GSM8K_PROMPTS = [
    "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?\nAnswer: Let's think step by step.",
    "Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nAnswer: Let's think step by step.",
    "Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\nAnswer: Let's think step by step.",
    "Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\nAnswer: Let's think step by step.",
    "Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. If the carry-over feed from the meals fed to the chickens in the morning and afternoon is 35 cups, how many cups of feed does she need to give her chickens in the final meal of the day if she has 20 chickens?\nAnswer: Let's think step by step.",
    "Question: Kylar went to the store to get water. He bought 6 gallons of water at $3 per gallon, 4 gallons at $5 per gallon, and a $2 bottle of water. How much money did he spend?\nAnswer: Let's think step by step.",
    "Question: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. If Seattle has 20 sheep, how many do Toulouse and Charleston have together?\nAnswer: Let's think step by step.",
    "Question: Carla is downloading a 200 GB file. She can download 2 GB/minute normally, but for every 40 GB she downloads, she has to pause for 10 minutes to reset her connection. How many minutes does she need to download the file?\nAnswer: Let's think step by step.",
    "Question: John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something at home. He tries to get home in 4 hours but spends the first 2 hours in standstill traffic. He spends the rest of the time driving at what speed in mph?\nAnswer: Let's think step by step.",
    "Question: Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?\nAnswer: Let's think step by step.",
]
# Pad to 50 with variations
while len(GSM8K_PROMPTS) < 50:
    i = len(GSM8K_PROMPTS) % 10
    GSM8K_PROMPTS.append(
        GSM8K_PROMPTS[i].replace("Let's think step by step.",
                                  f"Let's work through this carefully. (variant {len(GSM8K_PROMPTS)})")
    )

# 20 factual generation prompts
FACTUAL_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The speed of light in a vacuum is approximately",
    "The chemical formula for water is",
    "The largest planet in our solar system is",
    "The year the first moon landing occurred was",
    "The atomic number of carbon is",
    "Mount Everest is located on the border of",
    "The first president of the United States was",
    "DNA stands for",
    "The speed of sound in air at room temperature is approximately",
    "The periodic table was first published by",
    "Photosynthesis converts sunlight into",
    "The Great Wall of China was primarily built during",
    "Pi is approximately equal to",
    "The human body has approximately how many bones:",
    "The Pythagorean theorem states that",
    "The boiling point of nitrogen is approximately",
    "The speed of the Earth's rotation at the equator is about",
    "The distance from the Earth to the Sun is approximately",
]

# 20 deterministic continuation prompts
CONTINUATION_PROMPTS = [
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,",
    "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday,",
    "A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T,",
    "January, February, March, April, May, June, July, August,",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The quick brown fox jumped over the lazy dog. The quick brown fox",
    "for i in range(10):\n    print(i)\n\n# Output:\n# 0\n# 1\n# 2\n# 3\n# 4\n# 5\n#",
    "int main() {\n    printf(\"Hello, World!\\n\");\n    return",
    "Roses are red, violets are blue, sugar is sweet, and so are",
    "2 + 2 = 4\n3 + 3 = 6\n4 + 4 = 8\n5 + 5 = 10\n6 + 6 =",
    "The first ten prime numbers are: 2, 3, 5, 7, 11, 13, 17, 19, 23,",
    "Once upon a time, in a land far away, there lived a brave knight who",
    "H2O is water. NaCl is salt. CO2 is carbon dioxide. O2 is",
    "SELECT * FROM users WHERE age > 18 ORDER BY name",
    "In the beginning, God created the heavens and the earth. And the earth was",
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to",
    "E = mc^2, where E is energy, m is mass, and c is the speed of",
    "The Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55,",
    "HTTP/1.1 200 OK\nContent-Type: application/json\n\n{\"status\":",
    "import numpy as np\nimport pandas as pd\n\ndf = pd.DataFrame({",
]


def build_prompt_battery():
    """Build labeled prompt battery."""
    battery = []
    for i, p in enumerate(GSM8K_PROMPTS):
        battery.append({"category": "gsm8k", "index": i, "text": p})
    for i, p in enumerate(FACTUAL_PROMPTS):
        battery.append({"category": "factual", "index": i, "text": p})
    for i, p in enumerate(CONTINUATION_PROMPTS):
        battery.append({"category": "continuation", "index": i, "text": p})
    return battery


# ---------------------------------------------------------------------------
# Generation runner
# ---------------------------------------------------------------------------

def run_generation(model_name, kv_cache_dtype, prompts, max_tokens, gpu_mem):
    """Run generation and return per-prompt results with logprobs."""
    from vllm import LLM, SamplingParams

    extra_kwargs = {}
    if kv_cache_dtype == "int4_fused":
        extra_kwargs["enforce_eager"] = True
        gpu_mem = min(gpu_mem, 0.50)

    print(f"[drift] Loading model kv_cache_dtype={kv_cache_dtype} "
          f"gpu_mem={gpu_mem}...", file=sys.stderr)
    t0 = time.time()
    llm = LLM(
        model=model_name,
        dtype="float16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=2048,
        gpu_memory_utilization=gpu_mem,
        disable_log_stats=True,
        max_num_seqs=1,
        **extra_kwargs,
    )
    load_time = time.time() - t0
    print(f"[drift] Model loaded in {load_time:.1f}s", file=sys.stderr)

    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=10,  # top-10 logprobs per token
    )

    texts = [p["text"] for p in prompts]
    print(f"[drift] Generating {len(texts)} prompts x {max_tokens} tokens...",
          file=sys.stderr)
    t0 = time.time()
    outputs = llm.generate(texts, sp)
    gen_time = time.time() - t0
    print(f"[drift] Generation done in {gen_time:.1f}s", file=sys.stderr)

    results = []
    for prompt_info, output in zip(prompts, outputs):
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        # Extract logprobs: list of dicts mapping token_id -> logprob
        token_logprobs = []
        if completion.logprobs:
            for step in completion.logprobs:
                step_dict = {}
                for tok_id, logprob_obj in step.items():
                    step_dict[int(tok_id)] = {
                        "logprob": logprob_obj.logprob,
                        "rank": logprob_obj.rank,
                    }
                token_logprobs.append(step_dict)
        results.append({
            "category": prompt_info["category"],
            "index": prompt_info["index"],
            "token_ids": token_ids,
            "text": completion.text,
            "logprobs": token_logprobs,
        })

    # Cleanup GPU
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return results, load_time, gen_time


# ---------------------------------------------------------------------------
# Quant error measurement (standalone, per-head K/V analysis)
# ---------------------------------------------------------------------------

GROUP_SIZE = 32

def measure_quant_error_per_head(num_kv_heads=2, head_dim=128, seq_len=512,
                                  device="cuda"):
    """Measure K-side vs V-side quant-dequant error per head.

    Uses synthetic data drawn from a realistic distribution to characterize
    whether K or V is more sensitive to INT4 quantization.
    """
    torch.manual_seed(42)
    num_groups = head_dim // GROUP_SIZE

    # Generate synthetic K and V with realistic magnitude distributions
    # K tends to have sharper outliers than V in transformer models
    K = torch.randn(1, num_kv_heads, seq_len, head_dim,
                     dtype=torch.float16, device=device)
    V = torch.randn(1, num_kv_heads, seq_len, head_dim,
                     dtype=torch.float16, device=device)

    # Add realistic outliers to K (simulating attention key patterns)
    outlier_mask = torch.rand_like(K) < 0.01  # 1% outlier rate
    K[outlier_mask] *= 10.0

    results = {"k_error": [], "v_error": []}
    for h in range(num_kv_heads):
        k_head = K[0, h]  # [seq_len, head_dim]
        v_head = V[0, h]

        # Quantize and dequantize K
        k_reshaped = k_head.float().reshape(seq_len, num_groups, GROUP_SIZE)
        k_amax = k_reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        k_scale = k_amax / 7.0
        k_q = (k_reshaped / k_scale).round().clamp(-8, 7)
        k_deq = (k_q * k_scale).half().reshape(seq_len, head_dim)
        k_err = (k_head - k_deq).float().abs()

        # Quantize and dequantize V
        v_reshaped = v_head.float().reshape(seq_len, num_groups, GROUP_SIZE)
        v_amax = v_reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        v_scale = v_amax / 7.0
        v_q = (v_reshaped / v_scale).round().clamp(-8, 7)
        v_deq = (v_q * v_scale).half().reshape(seq_len, head_dim)
        v_err = (v_head - v_deq).float().abs()

        results["k_error"].append({
            "head": h,
            "mean": k_err.mean().item(),
            "max": k_err.max().item(),
            "p99": k_err.quantile(0.99).item(),
            "fraction_gt_01": (k_err > 0.1).float().mean().item(),
        })
        results["v_error"].append({
            "head": h,
            "mean": v_err.mean().item(),
            "max": v_err.max().item(),
            "p99": v_err.quantile(0.99).item(),
            "fraction_gt_01": (v_err > 0.1).float().mean().item(),
        })

    return results


# ---------------------------------------------------------------------------
# Token divergence analysis
# ---------------------------------------------------------------------------

def analyze_divergence(fp16_results, fused_results):
    """Compare FP16 vs fused generation token-by-token."""
    analyses = []
    for fp16_r, fused_r in zip(fp16_results, fused_results):
        fp16_ids = fp16_r["token_ids"]
        fused_ids = fused_r["token_ids"]
        fp16_lp = fp16_r["logprobs"]
        fused_lp = fused_r["logprobs"]

        # Find first divergent token
        first_div = None
        min_len = min(len(fp16_ids), len(fused_ids))
        for i in range(min_len):
            if fp16_ids[i] != fused_ids[i]:
                first_div = i
                break
        if first_div is None and len(fp16_ids) != len(fused_ids):
            first_div = min_len

        # Analyze divergence point
        div_analysis = None
        if first_div is not None and first_div < len(fp16_lp) and first_div < len(fused_lp):
            fp16_step = fp16_lp[first_div]
            fused_step = fused_lp[first_div]

            # Get top tokens from each at divergence point
            fp16_top = sorted(fp16_step.items(),
                              key=lambda x: x[1]["logprob"], reverse=True)[:5]
            fused_top = sorted(fused_step.items(),
                               key=lambda x: x[1]["logprob"], reverse=True)[:5]

            # Check if the fused top-1 appears in fp16's top-k
            fused_top1_id = fused_top[0][0] if fused_top else None
            fused_top1_in_fp16_topk = fused_top1_id in dict(fp16_top) if fused_top1_id else False

            # Rank of fused's choice in fp16's ranking
            fused_choice_fp16_rank = None
            if fused_top1_id and str(fused_top1_id) in fp16_step:
                fused_choice_fp16_rank = fp16_step[str(fused_top1_id)].get("rank")
            elif fused_top1_id and int(fused_top1_id) in fp16_step:
                fused_choice_fp16_rank = fp16_step[int(fused_top1_id)].get("rank")

            div_analysis = {
                "position": first_div,
                "fp16_token": fp16_ids[first_div],
                "fused_token": fused_ids[first_div],
                "fp16_top5": [(int(tid), info["logprob"]) for tid, info in fp16_top],
                "fused_top5": [(int(tid), info["logprob"]) for tid, info in fused_top],
                "fused_top1_in_fp16_top5": fused_top1_in_fp16_topk,
                "fused_choice_fp16_rank": fused_choice_fp16_rank,
            }

        # Count total matching tokens
        matching = sum(1 for i in range(min_len) if fp16_ids[i] == fused_ids[i])

        # Compute per-position agreement rate (for compounding analysis)
        per_pos_match = []
        for i in range(min_len):
            per_pos_match.append(1 if fp16_ids[i] == fused_ids[i] else 0)

        analyses.append({
            "category": fp16_r["category"],
            "index": fp16_r["index"],
            "fp16_len": len(fp16_ids),
            "fused_len": len(fused_ids),
            "first_divergent_token": first_div,
            "matching_tokens": matching,
            "total_compared": min_len,
            "match_rate": matching / min_len if min_len > 0 else 0,
            "divergence_analysis": div_analysis,
            "per_position_match": per_pos_match,
            "fp16_text_preview": fp16_r["text"][:200],
            "fused_text_preview": fused_r["text"][:200],
        })

    return analyses


def compute_summary(analyses):
    """Compute aggregate statistics across all prompts."""
    by_category = {}
    all_first_div = []
    all_match_rates = []

    for a in analyses:
        cat = a["category"]
        if cat not in by_category:
            by_category[cat] = {
                "count": 0,
                "diverged": 0,
                "first_div_positions": [],
                "match_rates": [],
            }
        by_category[cat]["count"] += 1
        all_match_rates.append(a["match_rate"])
        by_category[cat]["match_rates"].append(a["match_rate"])
        if a["first_divergent_token"] is not None:
            by_category[cat]["diverged"] += 1
            by_category[cat]["first_div_positions"].append(a["first_divergent_token"])
            all_first_div.append(a["first_divergent_token"])

    summary = {
        "total_prompts": len(analyses),
        "total_diverged": len(all_first_div),
        "overall_match_rate": sum(all_match_rates) / len(all_match_rates) if all_match_rates else 0,
    }

    if all_first_div:
        summary["first_div_median"] = sorted(all_first_div)[len(all_first_div) // 2]
        summary["first_div_mean"] = sum(all_first_div) / len(all_first_div)
        summary["first_div_min"] = min(all_first_div)
        summary["first_div_max"] = max(all_first_div)
        summary["immediate_divergence_count"] = sum(1 for d in all_first_div if d <= 1)
        summary["early_divergence_count"] = sum(1 for d in all_first_div if d <= 5)

    category_summary = {}
    for cat, info in by_category.items():
        cs = {
            "count": info["count"],
            "diverged": info["diverged"],
            "mean_match_rate": sum(info["match_rates"]) / len(info["match_rates"]) if info["match_rates"] else 0,
        }
        if info["first_div_positions"]:
            divs = sorted(info["first_div_positions"])
            cs["first_div_median"] = divs[len(divs) // 2]
            cs["first_div_mean"] = sum(divs) / len(divs)
        category_summary[cat] = cs
    summary["by_category"] = category_summary

    # Dominant failure signature
    if all_first_div:
        immediate = sum(1 for d in all_first_div if d <= 1)
        early = sum(1 for d in all_first_div if d <= 5)
        total = len(all_first_div)
        if immediate / total > 0.5:
            summary["dominant_signature"] = "IMMEDIATE_DIVERGENCE"
            summary["signature_detail"] = f"{immediate}/{total} prompts diverge at token 0-1"
        elif early / total > 0.5:
            summary["dominant_signature"] = "EARLY_DIVERGENCE"
            summary["signature_detail"] = f"{early}/{total} prompts diverge within first 5 tokens"
        else:
            summary["dominant_signature"] = "LATE_COMPOUNDING"
            summary["signature_detail"] = f"median first-divergent token = {summary['first_div_median']}"
    else:
        summary["dominant_signature"] = "NO_DIVERGENCE"
        summary["signature_detail"] = "FP16 and fused INT4 produce identical tokens"

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A0 generation-drift harness")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                        help="Model to test (default: Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens to generate per prompt (default: 200)")
    parser.add_argument("--gpu-mem", type=float, default=0.85,
                        help="GPU memory utilization for FP16 (default: 0.85)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: use subset of prompts (10 per category)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    battery = build_prompt_battery()
    if args.quick:
        # Use 10 per category for quick screening
        quick_battery = []
        for cat in ["gsm8k", "factual", "continuation"]:
            cat_prompts = [p for p in battery if p["category"] == cat][:10]
            quick_battery.extend(cat_prompts)
        battery = quick_battery

    print(f"[drift] Prompt battery: {len(battery)} prompts", file=sys.stderr)
    print(f"[drift] Model: {args.model}", file=sys.stderr)
    print(f"[drift] Max tokens: {args.max_tokens}", file=sys.stderr)

    # --- FP16 baseline ---
    print("\n=== FP16 BASELINE ===", file=sys.stderr)
    fp16_results, fp16_load, fp16_gen = run_generation(
        args.model, "auto", battery, args.max_tokens, args.gpu_mem)

    # --- Fused INT4 ---
    print("\n=== FUSED INT4 ===", file=sys.stderr)
    fused_results, fused_load, fused_gen = run_generation(
        args.model, "int4_fused", battery, args.max_tokens, args.gpu_mem)

    # --- Divergence analysis ---
    print("\n=== DIVERGENCE ANALYSIS ===", file=sys.stderr)
    analyses = analyze_divergence(fp16_results, fused_results)
    summary = compute_summary(analyses)

    # --- K/V quant error measurement ---
    print("\n=== K/V QUANT ERROR MEASUREMENT ===", file=sys.stderr)
    quant_errors = measure_quant_error_per_head(device="cuda")

    # --- Print summary to stderr ---
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"DRIFT HARNESS SUMMARY", file=sys.stderr)
    print(f"  Total prompts:       {summary['total_prompts']}", file=sys.stderr)
    print(f"  Total diverged:      {summary['total_diverged']}", file=sys.stderr)
    print(f"  Overall match rate:  {summary['overall_match_rate']:.4f}", file=sys.stderr)
    if 'first_div_median' in summary:
        print(f"  First div median:    {summary['first_div_median']}", file=sys.stderr)
        print(f"  First div mean:      {summary['first_div_mean']:.1f}", file=sys.stderr)
        print(f"  Immediate (<=1):     {summary.get('immediate_divergence_count', 0)}", file=sys.stderr)
        print(f"  Early (<=5):         {summary.get('early_divergence_count', 0)}", file=sys.stderr)
    print(f"  Dominant signature:  {summary['dominant_signature']}", file=sys.stderr)
    print(f"  Detail:              {summary['signature_detail']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    for cat, cs in summary.get("by_category", {}).items():
        print(f"  [{cat}] diverged={cs['diverged']}/{cs['count']} "
              f"match_rate={cs['mean_match_rate']:.4f}" +
              (f" first_div_median={cs.get('first_div_median', 'N/A')}" if cs['diverged'] > 0 else ""),
              file=sys.stderr)

    # --- Build output manifest ---
    manifest = {
        "harness": "fused_int4_drift_harness",
        "version": "1.0",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "prompt_count": len(battery),
        "kv_dtype_baseline": "auto",
        "kv_dtype_test": "int4_fused",
        "env": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "msl": os.environ.get("VLLM_FUSED_INT4_MIN_SEQ_LEN", "48"),
            "k_precision": os.environ.get("VLLM_FUSED_INT4_K_PRECISION", "int4"),
            "outlier_clip_ratio": os.environ.get("VLLM_FUSED_INT4_OUTLIER_CLIP_RATIO", "10.0"),
        },
        "timing": {
            "fp16_load_s": fp16_load,
            "fp16_gen_s": fp16_gen,
            "fused_load_s": fused_load,
            "fused_gen_s": fused_gen,
        },
        "summary": summary,
        "quant_error_per_head": quant_errors,
        "analyses": analyses,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output = json.dumps(manifest, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nResults written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
