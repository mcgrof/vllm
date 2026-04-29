# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CartridgeManifest: metadata for a single cartridge.

Every cartridge has a manifest that describes what model it was trained
for, its shape parameters, and optional routing/application metadata.
The manifest is used to:

1. Verify compatibility before loading (wrong model, wrong head count,
   wrong dtype = corrupted attention, garbage output, silent failure).
2. Index cartridges in a registry for multi-cartridge serving.
3. Carry routing labels (patient_id, doc_id, topic) for request routing.

The manifest can be stored as a JSON sidecar next to the .pt file, or
embedded in the checkpoint dict under a "manifest" key.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class CartridgeManifest:
    """Metadata for a single cartridge checkpoint."""

    # === Identity ===
    cartridge_id: str  # unique identifier (e.g., "patient_04_longhealth")

    # === Model compatibility ===
    model_id: str  # HuggingFace model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct")
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str  # "bfloat16", "float16", "float32"

    # === Shape ===
    num_tokens_raw: int  # total tokens before block alignment
    num_tokens_aligned: int  # tokens after align_to_block_size
    block_size: int
    num_blocks: int
    has_frozen_prefix: bool  # whether frozen_keys/values are present
    num_frozen_tokens: int = 0  # BOS/system tokens held fixed during training

    # === Integrity ===
    checksum: str = ""  # SHA-256 of the .pt file
    file_size_bytes: int = 0

    # === Provenance ===
    created_at: str = ""  # ISO 8601 timestamp
    training_steps: int = 0  # Self-Study training steps (0 = untrained prefill)
    source_document: str = ""  # description of what document was baked

    # === Routing labels (application-specific) ===
    labels: dict[str, str] = field(default_factory=dict)
    # e.g., {"patient_id": "patient_04", "doc_type": "medical_record",
    #        "topic": "pancreatic_cancer"}

    def validate_against_model(
        self,
        model_id: str,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> list[str]:
        """Check compatibility with a loaded model.

        Returns a list of mismatch descriptions. Empty list = compatible.
        """
        errors = []
        if self.model_id != model_id:
            errors.append(
                f"model_id mismatch: cartridge={self.model_id}, "
                f"server={model_id}"
            )
        if self.num_layers != num_layers:
            errors.append(
                f"num_layers mismatch: cartridge={self.num_layers}, "
                f"model={num_layers}"
            )
        if self.num_kv_heads != num_kv_heads:
            errors.append(
                f"num_kv_heads mismatch: cartridge={self.num_kv_heads}, "
                f"model={num_kv_heads}"
            )
        if self.head_dim != head_dim:
            errors.append(
                f"head_dim mismatch: cartridge={self.head_dim}, "
                f"model={head_dim}"
            )
        return errors

    def validate_against_block_size(self, block_size: int) -> list[str]:
        """Check that the cartridge was aligned to the server's block size."""
        errors = []
        if self.block_size != block_size:
            errors.append(
                f"block_size mismatch: cartridge={self.block_size}, "
                f"server={block_size}"
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CartridgeManifest:
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str | Path) -> CartridgeManifest:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_cartridge(
        cls,
        cartridge_path: str | Path,
        cartridge_id: str,
        model_id: str,
        block_size: int,
        labels: Optional[dict[str, str]] = None,
        training_steps: int = 0,
        source_document: str = "",
    ) -> CartridgeManifest:
        """Build a manifest by inspecting a cartridge .pt file.

        Loads the checkpoint to extract shape information, computes
        the file checksum, and constructs the manifest.
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
            align_to_block_size,
            load_cartridge,
        )

        path = Path(cartridge_path)
        cartridge = load_cartridge(str(path))

        # Compute checksum
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)

        # Detect frozen prefix
        checkpoint = torch.load(str(path), map_location="cpu",
                                weights_only=False)

        def _get(attr, obj):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            if isinstance(obj, dict) and attr in obj:
                return obj[attr]
            return None

        if hasattr(checkpoint, "trainable_keys"):
            cache = checkpoint
        elif isinstance(checkpoint, dict) and "cache" in checkpoint:
            cache = checkpoint["cache"]
        else:
            cache = checkpoint

        frozen_keys = _get("frozen_keys", cache)
        has_frozen = frozen_keys is not None and len(frozen_keys) > 0
        num_frozen = 0
        if has_frozen:
            fk = frozen_keys[0]
            if hasattr(fk, "data"):
                fk = fk.data
            num_frozen = fk.shape[2]  # (1, H, T_frozen, D)

        raw_tokens = cartridge["num_tokens"]
        aligned = align_to_block_size(raw_tokens, block_size)

        # Infer dtype from the first layer's K tensor
        first_k = None
        ckpt_for_dtype = torch.load(str(path), map_location="cpu",
                                     weights_only=False)
        tk = _get("trainable_keys", cache)
        if tk is not None and len(tk) > 0:
            t = tk[0]
            if hasattr(t, "data"):
                t = t.data
            first_k = t

        dtype_str = str(first_k.dtype).replace("torch.", "") if first_k is not None else "unknown"

        return cls(
            cartridge_id=cartridge_id,
            model_id=model_id,
            num_layers=cartridge["num_layers"],
            num_kv_heads=cartridge["num_kv_heads"],
            head_dim=cartridge["head_dim"],
            dtype=dtype_str,
            num_tokens_raw=raw_tokens,
            num_tokens_aligned=aligned,
            block_size=block_size,
            num_blocks=aligned // block_size,
            has_frozen_prefix=has_frozen,
            num_frozen_tokens=num_frozen,
            checksum=sha.hexdigest(),
            file_size_bytes=path.stat().st_size,
            training_steps=training_steps,
            source_document=source_document,
            labels=labels or {},
        )
