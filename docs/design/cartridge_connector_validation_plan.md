# CartridgeConnector Serving Validation Plan

**Scope:** This plan validates full-cartridge serving through vLLM's CartridgeConnector.
It does **not** validate sparse block routing, KRI routing, SPF, xa25, HKVD, or routing+repair research paths.
Those are separate research/evaluation branches.

## Tiers

- **Tier -1:** Static / unit validation (compileall, pytest, import checks)
- **Tier 0:** Single-cartridge smoke (vllm serve + 10 requests)
- **Tier 0.5:** Prefilled-cache equivalence canary
- **Tier 1:** HF-vs-vLLM quality parity (4 carts, 200 questions each)
- **Tier 2:** Multi-cartridge per-request dispatch (3 carts, isolation test)
- **Tier 3:** Qwen cross-model smoke/parity + wrong-model rejection
- **Tier 4:** GPU residency stress (LRU, refcount, memory stability)
- **Tier 5:** Concurrency / failure modes (16 concurrent, unknown ID, cancellation)
- **Tier 6:** TP / dtype smoke (TP=2, bf16/fp16/fp8)

See full plan at: /data/knlp-key-results/cartridge_test_plan_20260429.md
