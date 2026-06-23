# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resource identity and candidate abstractions for SPF.

Phase 1 of the honest-rewrite lands this module as the single source
of truth for "which reusable cache resource is this?". The older
code used a single ``prefix_hash: str`` which, combined with the
first-block-only convention that some callers used to compute it,
collapsed distinct prompts into the same identity whenever they
happened to share a first block (common system prompt, common chat
template prefix, etc.). That's the bug the ``false_shared_first_block``
test in ``tests/v1/core/test_spf_false_shared_first_block.py``
demonstrates on the legacy identity and that the new
:class:`ResourceId` fixes.

Contract of ``ResourceId.resource_id``:

  - It is a stable hash over the **entire reusable region** of the
    prompt (the full prefix span), not just the first block.
  - Two prompts with an identical first 128 tokens but different
    subsequent tokens MUST yield different ``resource_id`` values.
  - Identity is deterministic: same token sequence + same block
    size ⇒ same id, always.

``first_block_hash`` is retained on the candidate as a legacy field
so test code can still talk about "the old broken identity" and so
upstream integration code that only has the first block available
can still call into SPF, but it is explicitly not used for scoring
or outcome tracking in the honest-mode path.

Scopes: only ``PREFIX`` exists today. The enum is in place so future
scopes (session context, tool outputs, cartridge chunks) land as new
enum entries without breaking the identity contract.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Block size we hash token sequences into for prefix identities.
# 16 matches vLLM's default KV block size so a ResourceId covers an
# integer number of paged blocks.
DEFAULT_BLOCK_SIZE = 16


class ResourceScope(str, Enum):
    """Kinds of reusable cache resource SPF can identify.

    Phase-1 vanilla SPF only uses ``PREFIX``. Future extensions will
    add new scopes (e.g. ``SESSION_CONTEXT``, ``TOOL_OUTPUT``,
    ``CARTRIDGE_CHUNK``) as the SPF policy grows. Each scope is
    independently identifiable — two resources with different scopes
    are never conflated even if their byte content is identical.
    """

    PREFIX = "prefix"


def hash_prefix_tokens(token_ids: list[int]) -> str:
    """Stable hash over the full token sequence.

    Uses SHA-256 so the output is long enough that accidental
    collisions are not a practical concern during testing or
    deployment. Each token is encoded as 4 little-endian bytes.

    This is THE identity function for ``ResourceId.resource_id`` in
    scope ``PREFIX``. The choice is deliberately not first-block-only
    and never will be.
    """
    m = hashlib.sha256()
    for tid in token_ids:
        m.update(int(tid).to_bytes(4, "little", signed=False))
    return m.hexdigest()


def legacy_first_block_hash(
    token_ids: list[int],
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> str:
    """Compute the **legacy** first-block-only hash.

    Kept only so tests can construct the value the old identity
    convention used, and so the ``false_shared_first_block`` test
    can demonstrate that many distinct prompts sharing a first block
    all collapse to the same value under this convention while
    :func:`hash_prefix_tokens` distinguishes them correctly.

    DO NOT call this in production code paths. It is intentionally
    not exported from the ``spf`` package's ``__init__`` for exactly
    that reason.
    """
    first = list(token_ids)[:block_size]
    m = hashlib.sha256()
    m.update(b"legacy-first-block::")
    for tid in first:
        m.update(int(tid).to_bytes(4, "little", signed=False))
    return m.hexdigest()


@dataclass(frozen=True)
class ResourceId:
    """Stable identity for one reusable cache resource.

    ``resource_id`` is the load-bearing field. Other fields are
    metadata that the scoring / eviction policy uses without having
    to cross-reference elsewhere. ``first_block_hash`` is legacy
    compatibility only — it appears on the struct so upstream callers
    that only have the first block available can still construct a
    ResourceId, but the SPF scorer and outcome tracker use
    ``resource_id`` exclusively.
    """

    scope: ResourceScope
    resource_id: str
    num_blocks: int
    num_bytes: int
    session_id: str
    first_block_hash: Optional[str] = None

    def __post_init__(self):
        if self.num_blocks <= 0:
            raise ValueError(
                f"num_blocks must be > 0, got {self.num_blocks}"
            )
        if self.num_bytes < 0:
            raise ValueError(
                f"num_bytes must be >= 0, got {self.num_bytes}"
            )
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")
        if not isinstance(self.scope, ResourceScope):
            raise TypeError(
                f"scope must be ResourceScope, got {type(self.scope)}"
            )

    @classmethod
    def for_prefix(
        cls,
        token_ids: list[int],
        session_id: str,
        block_size: int = DEFAULT_BLOCK_SIZE,
        bytes_per_token: int = 0,
    ) -> "ResourceId":
        """Build a PREFIX-scope ResourceId from a token sequence.

        ``num_blocks`` is computed by block-aligned ceiling of the
        token count (each ResourceId covers an integer number of
        paged blocks). ``num_bytes`` is an optional size hint that
        callers can supply when they know the KV footprint; it is
        not required for identity or scoring in phase 1.
        """
        if not token_ids:
            raise ValueError(
                "PREFIX resource requires at least one token")
        n = len(token_ids)
        nb = (n + block_size - 1) // block_size
        return cls(
            scope=ResourceScope.PREFIX,
            resource_id=hash_prefix_tokens(token_ids),
            num_blocks=nb,
            num_bytes=bytes_per_token * n,
            session_id=session_id,
            first_block_hash=legacy_first_block_hash(
                token_ids, block_size=block_size),
        )

    @property
    def short_id(self) -> str:
        """First 8 hex chars of resource_id, for log readability."""
        return self.resource_id[:8]


@dataclass
class ResourceCandidate:
    """A candidate resource under consideration by the SPF policy.

    This is the object the scorer ranks and the one outcome tracking
    is tied to. ``last_query_hash`` is included for future priors
    (KRI-Q and similar) but the vanilla-SPF scoring path does NOT
    consult it — phase-1 rule "no routing priors yet" is enforced by
    not passing the field to the scorer.
    """

    resource: ResourceId
    score: float = 0.0
    last_query_hash: Optional[str] = None
    # Ordered context the scorer may use: transition source, burst
    # state, etc. Populated by the controller, not the caller.
    ctx: dict = field(default_factory=dict)

    @property
    def resource_id(self) -> str:
        """Short alias: the load-bearing hash."""
        return self.resource.resource_id

    @property
    def session_id(self) -> str:
        return self.resource.session_id

    @property
    def num_blocks(self) -> int:
        return self.resource.num_blocks

    @property
    def num_bytes(self) -> int:
        return self.resource.num_bytes

    @property
    def scope(self) -> ResourceScope:
        return self.resource.scope
