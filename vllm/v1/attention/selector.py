# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from typing import NamedTuple, cast, get_args

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.registry import (
    MAMBA_TYPE_TO_BACKEND_MAP,
    MambaAttentionBackendEnum,
)

logger = init_logger(__name__)


class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER

    def __repr__(self):
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "
            f"dtype={self.dtype}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"block_size={self.block_size}, "
            f"use_mla={self.use_mla}, "
            f"has_sink={self.has_sink}, "
            f"use_sparse={self.use_sparse}, "
            f"use_mm_prefix={self.use_mm_prefix}, "
            f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "
            f"attn_type={self.attn_type})"
        )


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    is_asymmetric = False
    if kv_cache_dtype is not None:
        # Asymmetric K/V: if the spec is a tuple, validate each
        # element and reduce to the K dtype for selector purposes.
        if isinstance(kv_cache_dtype, tuple):
            valid = get_args(CacheDType)
            for dt in kv_cache_dtype:
                assert dt in valid, (
                    f"Invalid dtype in asymmetric spec: {dt}")
            # Reducing to the key dtype makes an asymmetric cache look
            # symmetric to the selector, so a 16-bit key with an 8-bit
            # value reads as plain 16-bit and can be handed to a backend
            # that writes the cache with a single dtype. That backend
            # then rejects the pair at the first write, after the model
            # has loaded. Remember that it was a pair.
            is_asymmetric = kv_cache_dtype[0] != kv_cache_dtype[1]
            kv_cache_dtype = kv_cache_dtype[0]  # K dtype
        else:
            valid_cache_dtypes = get_args(CacheDType)
            assert kv_cache_dtype in valid_cache_dtypes, (
                f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
                f"Valid values are: {valid_cache_dtypes}"
            )

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    cache_config = vllm_config.cache_config
    if cache_config is not None and cache_config.user_specified_block_size:
        block_size = cache_config.block_size
    else:
        block_size = None

    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type or AttentionType.DECODER,
    )

    requested = vllm_config.attention_config.backend
    from vllm import envs as _envs

    if getattr(_envs, "VLLM_PREBIAS_K", False) and not use_mla:
        # The model subtracts the key bias before caching and only the
        # FlashInfer kernel reconstructs it. Any other backend serves
        # attention with the bias missing and reports nothing wrong, so
        # this cannot be left to the default choice.
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if requested is None:
            requested = AttentionBackendEnum.FLASHINFER
            logger.info(
                "VLLM_PREBIAS_K is set: selecting the %s backend, the "
                "only one that reconstructs the subtracted key bias.",
                requested.name,
            )
        elif requested != AttentionBackendEnum.FLASHINFER:
            raise ValueError(
                f"VLLM_PREBIAS_K is set but the {requested.name} backend "
                "was requested. Only FLASHINFER reconstructs the key bias "
                "the model subtracts; any other backend would serve wrong "
                "attention silently. Unset VLLM_PREBIAS_K or request "
                "FLASHINFER."
            )

    if is_asymmetric and requested is None:
        # Writing a key and a value at different dtypes takes two calls,
        # which only some backends do. Choosing by the key dtype alone
        # picks one that cannot, so name the one that can rather than
        # failing at the first cache write.
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        requested = AttentionBackendEnum.FLASHINFER
        logger.info(
            "Asymmetric key/value cache dtypes %s: selecting the "
            "%s backend, which writes the halves separately.",
            attn_selector_config.kv_cache_dtype,
            requested.name,
        )

    return _cached_get_attn_backend(
        backend=requested,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


@cache
def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    from vllm.platforms import current_platform

    attention_cls = current_platform.get_attn_backend_cls(
        backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    requested_name = backend.name if backend is not None else "auto"
    backend = resolve_obj_by_qualname(attention_cls)

    # Backend verification logging — machine-readable for benchmark manifests
    logger.info(
        "Backend manifest: requested_backend=%s, selected_backend=%s, "
        "kv_cache_dtype=%s, head_size=%s, dtype=%s",
        requested_name,
        backend.get_name(),
        attn_selector_config.kv_cache_dtype,
        attn_selector_config.head_size,
        attn_selector_config.dtype,
    )

    # Adjust kv cache layout if the selected backend requires a specific one
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout(required_layout)
        logger.info(
            "Using %s KV cache layout for %s backend.",
            required_layout,
            backend.get_name(),
        )

    return backend


def get_mamba_attn_backend(
    mamba_type: str,
) -> type[AttentionBackend]:
    """Select which mamba attention backend to use and lazily import it."""
    return _cached_get_mamba_attn_backend(mamba_type)


@cache
def _cached_get_mamba_attn_backend(
    mamba_type: str,
) -> type[AttentionBackend]:
    assert mamba_type and isinstance(mamba_type, str)

    selected_backend = None
    try:
        backend_name = MAMBA_TYPE_TO_BACKEND_MAP[mamba_type]
        selected_backend = MambaAttentionBackendEnum[backend_name]
    except KeyError as e:
        raise ValueError(
            f"Invalid mamba attention backend type: '{backend_name}'. Valid "
            f"backends are: {list(MambaAttentionBackendEnum.__members__.keys())}"
        ) from e

    mamba_attn_backend = selected_backend.get_class()
    return mamba_attn_backend
