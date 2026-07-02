# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import field
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.torch_utils import (
    is_quantized_kv_cache,
    kv_cache_uses_per_token_head_scales,
)

logger = init_logger(__name__)

CacheDType = Literal[
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
]
MambaDType = Literal["auto", "float32", "float16", "bfloat16"]
MambaCacheMode = Literal["all", "align", "none"]
PrefixCachingHashAlgo = Literal["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"]
KVOffloadingBackend = Literal["native", "lmcache"]

# Asymmetric K/V: cache_dtype can be a single string (symmetric,
# backward compatible) or a 2-tuple of strings (K dtype, V dtype).
# Example: ("float16", "fp8_e4m3") keeps keys at FP16 while storing
# values in FP8.
CacheDTypeSpec = CacheDType | tuple[CacheDType, CacheDType]


def parse_cache_dtype_spec(raw: str) -> CacheDTypeSpec:
    """Parse a cache dtype spec from a CLI string.

    Accepts either a single token ("fp8_e4m3") or a pair separated by
    a comma or a pipe ("float16,fp8_e4m3" / "float16|fp8_e4m3"). The
    pipe form exists for callers whose own argument parsing splits on
    commas (e.g. lm-eval --model_args). Returns either a CacheDType
    string or a (k_dtype, v_dtype) tuple.
    """
    for sep in (",", "|"):
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep, 1)]
            return (parts[0], parts[1])  # type: ignore[return-value]
    return raw  # type: ignore[return-value]


def cache_dtype_k(spec: CacheDTypeSpec) -> CacheDType:
    """Extract the K dtype from a CacheDTypeSpec."""
    if isinstance(spec, tuple):
        return spec[0]
    return spec


def cache_dtype_v(spec: CacheDTypeSpec) -> CacheDType:
    """Extract the V dtype from a CacheDTypeSpec."""
    if isinstance(spec, tuple):
        return spec[1]
    return spec


def is_asymmetric_kv(spec: CacheDTypeSpec) -> bool:
    """True if K and V use different dtypes."""
    if isinstance(spec, tuple):
        return spec[0] != spec[1]
    return False


@config
class CacheConfig:
    """Configuration for the KV cache."""

    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16

    block_size: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """Size of a contiguous cache block in number of tokens.
    Accepts None (meaning "use default"). After construction, always int."""
    user_specified_block_size: bool = field(default=False, init=False)
    """Whether block_size was explicitly provided. Derived automatically."""
    user_specified_mamba_block_size: bool = field(default=False, init=False)
    """Whether mamba_block_size was explicitly provided. Derived automatically."""
    hash_block_size: int | None = Field(default=None, gt=0)
    """Block size (in tokens) used for computing Request's block_hashes.

    This can be set to a finer granularity than the physical KV cache block
    sizes (e.g. 8) as long as every KV cache group's `block_size` is divisible
    by it. This enables prefix-caching keys to be computed at the finest common
    granularity and then merged for larger physical block sizes.

    This config is not static default. If left unspecified, vLLM will choose a
    default based on the resolved KV cache groups (typically the smallest KV
    cache block size when there are multiple groups).
    """
    gpu_memory_utilization: float = Field(default=0.92, gt=0, le=1)
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.92. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""
    cache_dtype: CacheDTypeSpec = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc).
    Some models (namely DeepSeekV3.2) default to fp8, set to bfloat16 to use
    bfloat16 instead, this is an invalid option for models that do not default
    to fp8.
    For asymmetric K/V, pass a comma-separated or pipe-separated pair via CLI:
    --kv-cache-dtype float16,fp8_e4m3 (FP16 keys, FP8 values).
    """
    v_cache_dtype: str | None = None
    """V-side dtype string for asymmetric K/V. Mirrors the V half of
    ``cache_dtype`` when the spec is a tuple (e.g. "fp8_e4m3"). Read by
    ``vllm/v1/worker/gpu/attn_utils.py`` to inject ``v_dtype`` into the
    per-layer ``AttentionSpec`` so the asymmetric tuple-cache split in
    ``_reshape_kv_cache`` triggers. ``None`` on the symmetric path.

    Wired from the CLI via ``EngineArgs.kv_cache_dtype`` after the
    tuple parse in ``arg_utils.py``."""
    is_attention_free: bool = False
    """Whether the model is attention-free. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    num_gpu_blocks_override: int | None = None
    """Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
    if specified. Does nothing if `None`. Used for testing preemption."""
    sliding_window: int | None = None
    """Sliding window size for the KV cache. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    enable_prefix_caching: bool = True
    """Whether to enable prefix caching."""
    prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"
    """Set the hash algorithm for prefix caching:

    - "sha256" uses Pickle for object serialization before hashing. This is the current
      default, as SHA256 is the most secure choice to avoid potential hash collisions.
    - "sha256_cbor" provides a reproducible, cross-language compatible hash. It
      serializes objects using canonical CBOR and hashes them with SHA-256.
    - "xxhash" uses Pickle serialization with xxHash (128-bit) for faster,
      non-cryptographic hashing. Requires the optional ``xxhash`` package.
      IMPORTANT: Use of a hashing algorithm that is not considered  cryptographically
      secure theoretically increases the risk of hash collisions, which can cause
      undefined behavior or even leak private information in multi-tenant environments.
      Even if collisions are still very unlikely, it is important to consider your
      security risk tolerance against the performance benefits before turning this on.
    - "xxhash_cbor" combines canonical CBOR serialization with xxHash for
      reproducible hashing. Requires the optional ``xxhash`` package."""
    calculate_kv_scales: bool = False
    """Deprecated: This option is deprecated and will be removed in v0.19.
    It enables dynamic calculation of `k_scale` and `v_scale` when
    kv_cache_dtype is fp8. If `False`, the scales will be loaded from the model
    checkpoint if available. Otherwise, the scales will default to 1.0."""
    kv_cache_dtype_skip_layers: list[str] = field(default_factory=list)
    """Layer patterns to skip KV cache quantization. Accepts layer indices
    (e.g., '0', '2', '4') or attention type names (e.g., 'sliding_window')."""
    mamba_page_size_padded: int | None = None
    """ Optional override for mamba page size; used by hybrid mamba/attention
    models to ensure exact alignment with attention page size."""
    mamba_block_size: int | None = Field(default=None, gt=0)
    """Size of a contiguous cache block in number of tokens for mamba cache.
    Can be set only when prefix caching is enabled.
    Value must be a multiple of 8 to align with causal_conv1d kernel."""
    mamba_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (both the conv as well as the
    ssm state). If set to 'auto', the data type will be inferred from the model
    config."""
    mamba_ssm_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (ssm state only, conv state will
    still be controlled by mamba_cache_dtype). If set to 'auto', the data type
    for the ssm state will be determined by mamba_cache_dtype."""
    mamba_cache_mode: MambaCacheMode = "none"
    """The cache strategy for Mamba layers.
    - "none": set when prefix caching is disabled.
    - "all": cache the mamba state of all tokens at position i * block_size. This is
           the default behavior (for models that support it) when prefix caching is
           enabled.
    - "align": only cache the mamba state of the last token of each scheduler step and
           when the token is at position i * block_size.
    """

    # Will be set after profiling.
    num_gpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    num_cpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""

    # Set after KV cache initialization.
    kv_cache_size_tokens: int | None = field(default=None, init=False)
    """Per-DP-engine KV cache capacity in tokens (group-aware). Uses
    group-aware capacity since num_gpu_blocks * block_size can be wrong
    for hybrid models where requests occupy multiple KV cache groups."""
    kv_cache_max_concurrency: float | None = field(default=None, init=False)
    """Per-DP-engine maximum concurrency at max_model_len tokens."""

    kv_sharing_fast_prefill: bool = False
    """This feature is work in progress and no prefill optimization takes place
    with this flag enabled currently.

    In some KV sharing setups, e.g. YOCO (https://arxiv.org/abs/2405.05254),
    some layers can skip tokens corresponding to prefill. This flag enables
    attention metadata for eligible layers to be overridden with metadata
    necessary for implementing this optimization in some models (e.g. Gemma3n)
    """

    kv_cache_memory_bytes: int | None = None
    """Size of KV Cache per GPU in bytes. By default, this is set to None
    and vllm can automatically infer the kv cache size based on
    gpu_memory_utilization. However, users may want to manually specify
    the kv cache memory size. kv_cache_memory_bytes allows more fine-grain
    control of how much memory gets used when compared with using
    gpu_memory_utilization. Note that kv_cache_memory_bytes
    (when not-None) ignores gpu_memory_utilization"""

    kv_offloading_size: float | None = None
    """Size of the KV cache offloading buffer in GiB. When TP > 1, this is
    the total buffer size summed across all TP ranks. By default, this is set
    to None, which means no KV offloading is enabled. When set, vLLM will
    enable KV cache offloading to CPU using the kv_offloading_backend."""

    kv_offloading_backend: KVOffloadingBackend = "native"
    """The backend to use for KV cache offloading. Supported backends include
    'native' (vLLM native CPU offloading), 'lmcache'.
    KV offloading is only activated when kv_offloading_size is set."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        ignored_factors = {
            # Runtime/derived knobs that don't affect compiled graph shape
            "gpu_memory_utilization",
            "is_attention_free",
            "num_gpu_blocks_override",
            "enable_prefix_caching",
            "prefix_caching_hash_algo",
            # Prefix-caching implementation detail (doesn't affect compiled graph).
            "hash_block_size",
            "mamba_page_size_padded",
            "user_specified_block_size",
            "user_specified_mamba_block_size",
            "_block_size_resolved",
            # Post-init/derived counters
            "num_gpu_blocks",
            "num_cpu_blocks",
            "kv_cache_size_tokens",
            "kv_cache_max_concurrency",
            # WIP feature toggle not impacting compiled graph shape
            "kv_sharing_fast_prefill",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def metrics_info(self):
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    _block_size_resolved: bool = field(default=False, init=False)
    """Guard against pydantic re-running _apply_block_size_default."""

    @field_validator("block_size", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        if value is None:
            return value
        return handler(value)

    @model_validator(mode="after")
    def _apply_block_size_default(self) -> "CacheConfig":
        # Pydantic re-runs validators when CacheConfig is nested inside
        # another pydantic model (e.g. VllmConfig). Guard against that.
        if self._block_size_resolved:
            return self
        self._block_size_resolved = True
        if self.block_size is None:
            self.block_size = self.DEFAULT_BLOCK_SIZE
        else:
            self.user_specified_block_size = True
        if self.mamba_block_size is not None:
            self.user_specified_mamba_block_size = True
        return self

    @field_validator("calculate_kv_scales", mode="after")
    @classmethod
    def _warn_deprecated_calculate_kv_scales(cls, calculate_kv_scales: bool) -> bool:
        if calculate_kv_scales:
            logger.warning(
                "The `--calculate-kv-scales` option is deprecated and will "
                "be removed in v0.19. The scales will be loaded from the "
                "model checkpoint if available, otherwise they default to "
                "1.0."
            )
        return calculate_kv_scales

    @field_validator("cache_dtype", mode="after")
    @classmethod
    def _validate_cache_dtype(
        cls,
        cache_dtype: CacheDTypeSpec,
    ) -> CacheDTypeSpec:
        if isinstance(cache_dtype, tuple):
            k_dt, v_dt = cache_dtype
            if k_dt == v_dt:
                # Degenerate pair: normalize to the single-string form
                # so downstream str-typed call sites see the symmetric
                # spec they expect.
                cache_dtype = k_dt
            else:
                if k_dt not in ("auto", "float16", "bfloat16"):
                    raise ValueError(
                        "Asymmetric KV cache requires an unquantized K "
                        f"dtype (auto, float16 or bfloat16); got K={k_dt!r}."
                    )
                if not v_dt.startswith("fp8"):
                    raise ValueError(
                        f"Asymmetric KV cache requires an FP8 V dtype; got V={v_dt!r}."
                    )
                logger.info("Using asymmetric KV cache dtypes: K=%s, V=%s.", k_dt, v_dt)
                return cache_dtype
        if kv_cache_uses_per_token_head_scales(cache_dtype):
            logger.info(
                "Using %s data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Dynamic per-token-head scales will be computed at runtime.",
                str(cache_dtype),
            )
        elif is_quantized_kv_cache(cache_dtype):
            logger.info(
                "Using %s data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor",
                str(cache_dtype),
            )
        return cache_dtype
