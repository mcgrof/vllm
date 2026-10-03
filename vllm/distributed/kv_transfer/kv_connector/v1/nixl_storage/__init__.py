# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage-mediated prefill/decode handoff through NIXL.

The prefill worker stages a finished request's KV blocks to host memory,
writes them to a local storage target through NIXL's IO_URING backend and
publishes a manifest; the decode worker reads the extents back into host
memory, restores them into its GPU KV cache and acknowledges. A same-host
coordinator owns the extents and the manifest state machine. Ordinary
memory-to-memory NIXL P/D is untouched and remains the reference.
"""
