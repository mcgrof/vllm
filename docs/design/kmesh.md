# TokenDance-style shared-update recovery

This experimental integration tests whether agent prompts can reuse the same
update after different private histories. Ordinary prefix caching stops at the
first differing token. KMesh instead carries explicit token spans and groups
compatible requests whose declared shared spans contain identical token IDs.
The shared name alone never establishes content identity or safe numerical reuse.

The implemented baseline consists of request metadata, same-step planning,
serial and collective position-independent recovery through LMCache, and offline
Master–Mirror encoding. It keeps all live cache destinations request-owned.
The branch contains no DMA-BUF storage integration, speculative prefetch policy,
custom scheduling barrier, or physical GPU-page aliasing implementation.

## Runtime contract

Pass primitive metadata in `SamplingParams.extra_args["kmesh"]`. The boundary
parser accepts explicit sideband spans or reserved separators and produces
immutable request context. The planner checks token content, execution and round,
trust domain, model/revision, adapter, dtype, layout, parallelism, attention backend
and active recovery window before creating a bounded same-step group.

The KMesh connector sends an immutable admission-generation descriptor to the
LMCache connector through `MultiConnector`. The worker receives an exact step
plan. Lookup, load, model execution and cleanup retain separate ownership; a
failed load suppresses external handoff before local recomputation. Abort,
preemption and replacement cannot authorize reuse through a stale descriptor.
Ordinary connectors inherit no-op lifecycle and peer-metadata hooks.

Supported target profiles are:

| Profile | Execution |
| --- | --- |
| `serial_pic_v1` | Request-local position-independent cache recovery |
| `collective_lockstep_v1` | Independent sessions advanced in layer order |
| `collective_batched_position_recovery_v1` | One cached-key position-recovery call per eligible group/layer |
| `collective_batched_qkv_v1` | Also batch query/key rotary encoding and check-layer key differences |

Query/key/value projections and attention remain request-local. Important-position
selection, gap repair, masks and destination pages also remain request-local.
A collective operation failure invalidates its participating requests; misses
remain independent. Per-member collective counters count participation in shared
calls and must not be summed as a count of unique group calls.

## Qualification boundary

The retained GPU evidence uses Qwen2.5-0.5B-Instruct at model revision
`7ae557604adf67be50417f59c2c2f167def9a775`, BF16, the Triton attention backend,
eager execution, synchronous scheduling, and one device with tensor/pipeline
parallel sizes of one. The executable profile is intentionally pinned, including
the model snapshot path in `vllm/v1/kmesh/config.py`.

The compatible source bases are vLLM `454ea5b52611c933e00581723e2db56f0144cea7`
and LMCache `499c830b3fda473388a2b7d6bb002dd69f5d7867`. Pair this branch with
LMCache's `2026-10-16-feature-tokendance-reproduction` branch. Newer prefetch or
storage branches implement a different integration and are not substitutes.

Numerical parity is against the serial recovery path. That approximate path is
not established as elementwise equivalent to full prompt recomputation. No
throughput, latency, live-memory capacity, or end-to-end paper reproduction
claim follows from these sources. The retained GPU results are inherited;
creating a dated branch does not rerun their experiments.

## Reproduction components

[The benchmark guide](../../benchmarks/kmesh/README.md) gives deterministic
synthetic trace generation and ordinary/observer controls. Those commands do not
execute serial or collective recovery. The strict eight-arm GPU experiment driver
and verifier are separate archived experiment inputs. Their source-identity pins
must be updated for a new series before a GPU rerun; do not bypass those checks.
They are not included in this source branch.

`tests/v1/kmesh/` covers parsing, planning, profiles, handoff and codec contracts.
LMCache supplies session, lookup, allocation, interruption, numerical-operation
and terminal-cleanup tests. CPU rotary doubles exercise real tensor operations
but do not qualify the GPU runtime.

The vLLM offline codec compares position-aligned blocks. LMCache's opt-in family
serializer aligns one shared span per member by content position and stores
private tokens plus exact differences. These are distinct artifact codecs;
encoded byte counts include metadata and never imply shared live GPU pages.
