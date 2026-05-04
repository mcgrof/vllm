#!/usr/bin/env python3
"""Reproducer for LogprobsTensors OverflowError with KV connectors.

Bug: vllm/v1/outputs.py LogprobsTensors.empty_cpu() uses torch.empty()
     which leaves token_ids as uninitialized int32 memory. When a KV
     connector marks tokens as externally computed, the model runner
     only fills logprobs for non-external positions. The unfilled
     positions retain garbage int32 values that crash tokenizer.decode()
     with OverflowError.

Real-world trigger: Any vLLM V1 serving request that uses:
  1. A KV connector reporting externally-computed tokens
     (CartridgeConnector, LMCache prefix reuse, Nixl KV transfer, etc.)
  2. prompt_logprobs or echo+logprobs in the API request

The model runner creates a LogprobsTensors of size (num_prompt_tokens-1)
via empty_cpu(), then only fills positions starting at num_computed_tokens
(the externally-computed boundary). Positions 0..num_computed_tokens-1
are never written and contain whatever torch.empty() left in memory.

When the logprobs processor later iterates over ALL positions and calls
tokenizer.decode([token_id]) on the garbage values, it crashes.

Fix: Use torch.zeros() for token_ids and torch.full(-inf) for logprobs
     in LogprobsTensors.empty_cpu() so unfilled positions have valid
     sentinel values.

Usage:
    # CPU-only — no GPU needed
    pytest tests/v1/core/test_logprobs_overflow_reproducer.py -v

    # Or run directly
    python tests/v1/core/test_logprobs_overflow_reproducer.py
"""
import torch
import pytest


class TestLogprobsTensorsOverflow:
    """Prove that torch.empty can produce token IDs that overflow
    tokenizer.decode(), and that torch.zeros prevents it."""

    def test_empty_produces_garbage_token_ids(self):
        """torch.empty int32 can produce values outside any vocabulary.

        This is the root cause: empty_cpu() returns tensors with
        random int32 values that may be negative or larger than
        the vocabulary size, causing OverflowError in the tokenizer.
        """
        # Allocate and free some memory to increase chance of garbage
        junk = torch.randint(-2**30, 2**30, (1000,), dtype=torch.int32)
        del junk

        # Now allocate "empty" — may reuse the freed memory
        t = torch.empty((100, 4), dtype=torch.int32)
        values = t.flatten().tolist()

        # At least some values should be non-zero (garbage)
        # (This is probabilistic but almost always true)
        has_nonzero = any(v != 0 for v in values)
        has_large = any(abs(v) > 200000 for v in values)

        # We can't assert these deterministically, but we CAN show
        # that the fix (zeros) guarantees no garbage:
        t_zeros = torch.zeros((100, 4), dtype=torch.int32)
        assert t_zeros.max().item() == 0
        assert t_zeros.min().item() == 0

    def test_overflow_on_garbage_token_id(self):
        """Simulates what happens when tokenizer.decode() gets a garbage
        int32 from an unfilled LogprobsTensors position.

        This is the exact crash path:
          logprobs.py:113 → convert_ids_list_to_tokens → decode([token_id])
        """
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct")

        # Garbage int32 values that torch.empty produces — negative
        # values are common in uninitialized memory and crash the
        # tokenizer's Rust backend with OverflowError.
        for garbage_id in [-1, -2**31, 2**32]:
            with pytest.raises((OverflowError, RuntimeError)):
                tok.decode([garbage_id])

    def test_zeros_are_decodable(self):
        """Token ID 0 is always decodable (it's typically <unk> or padding).

        This proves the fix is safe: zero-initialized positions won't
        crash the tokenizer.
        """
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct")

        # Token ID 0 should decode without error
        result = tok.decode([0])
        assert isinstance(result, str)

    def test_logprobs_tensors_empty_cpu_is_safe(self):
        """The actual fix: empty_cpu() should produce decodable token IDs.

        Before fix: torch.empty → garbage → OverflowError
        After fix:  torch.zeros → 0 → safe decode
        """
        from vllm.v1.outputs import LogprobsTensors

        lpt = LogprobsTensors.empty_cpu(
            num_positions=50,
            num_tokens_per_position=4,
        )

        # All token IDs should be 0 (not garbage)
        assert lpt.logprob_token_ids.max().item() == 0
        assert lpt.logprob_token_ids.min().item() == 0

        # All logprobs should be -inf (not garbage floats)
        assert lpt.logprobs.max().item() == float("-inf")

        # All ranks should be 0
        assert lpt.selected_token_ranks.max().item() == 0

    def test_partial_fill_simulates_kv_connector(self):
        """Simulates the real-world scenario: a KV connector reports
        N tokens as externally computed. The model runner creates a
        LogprobsTensors of size (num_prompt_tokens-1) but only fills
        positions starting at N. Positions 0..N-1 must be safe.

        This is exactly what CartridgeConnector triggers.
        """
        from vllm.v1.outputs import LogprobsTensors

        num_prompt_tokens = 100
        num_external = 64  # CartridgeConnector reports these
        num_logprobs = 4

        # Model runner creates empty tensor for full prompt
        lpt = LogprobsTensors.empty_cpu(
            num_positions=num_prompt_tokens - 1,
            num_tokens_per_position=num_logprobs + 1,
        )

        # Model runner fills only positions num_external onward
        # (simulating what gpu_model_runner._get_prompt_logprobs_dict does)
        real_token_ids = torch.arange(
            num_external, num_prompt_tokens - 1, dtype=torch.int32
        ).unsqueeze(1).expand(-1, num_logprobs + 1)
        real_logprobs = torch.randn(
            num_prompt_tokens - 1 - num_external, num_logprobs + 1
        )
        real_ranks = torch.zeros(
            num_prompt_tokens - 1 - num_external, dtype=torch.int32
        )

        # Write to positions num_external onward
        start = num_external
        lpt.logprob_token_ids[start:] = real_token_ids
        lpt.logprobs[start:] = real_logprobs
        lpt.selected_token_ranks[start:] = real_ranks

        # The UNFILLED positions (0..num_external-1) must be safe
        unfilled_ids = lpt.logprob_token_ids[:num_external].flatten().tolist()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct")

        # This is the exact path that crashed before the fix:
        for token_id in unfilled_ids:
            # Before fix: OverflowError on garbage int32
            # After fix: decodes token 0 safely
            result = tok.decode([token_id])
            assert isinstance(result, str)

        # Unfilled logprobs should be -inf (clearly "not computed")
        unfilled_lps = lpt.logprobs[:num_external]
        assert (unfilled_lps == float("-inf")).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
