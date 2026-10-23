# KMesh multiprocess handoff admission

Status: proposed engine boundary. The existing in-process position-recovery
profiles are separate from ordinary multiprocess LMCache prefix caching.

Two prompts can contain the same update after different private histories.
KMesh can describe that update explicitly, but recognizing an MP connector name
does not prove that its server receives the span, restores it at the right
position, or retains its destination until device work completes. Inherited
no-op connector hooks cannot provide that proof.

Allow the exact named MP child through configuration inspection, preserving
every other target-profile restriction. Resolve its actual class through the
normal connector module, including the external LMCache implementation when
selected. Reject substituted classes. Before composite construction completes
and before KMesh arms, require the LMCache child to expose and pass a public
handoff validator. A missing or failed validator must propagate through normal
construction cleanup and must not arm KMesh.

The builtin MP connector must reject every explicit KMesh profile before base
initialization or adapter construction. Its handoff validator must also refuse
when called on an unprofiled instance. The corresponding external LMCache
implementation needs the same boundary. Ordinary unprofiled MP operation must
continue independently.

Qualification must cover exact-class resolution, missing and failed handoff
proof, proof before arming, both scheduler and worker construction, and the
existing in-process connector controls. A future positive MP contract requires
server/domain/context identity, explicit-span execution, and actual completion
and exporter lifetime evidence. CPU admission checks do not establish those
properties, GPU correctness, model parity, or a storage-latency improvement.
