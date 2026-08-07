# Compatibility levels

Compatibility class describes how an engine bundle relates to the GPU that runs it. It is separate from the container image.

## Exact

Exact means the bundle was built and validated for the named target profile. This is the primary evidence mode.

Current examples:

```text
rtx4070-ubuntu2404
rtx5060-laptop-ubuntu2404
orin-nano-l4t3647
```

The staged target needs the future
`rtx3050-laptop-ubuntu2404` profile; it does not exist yet.

## Same compute capability

A same-compute bundle is grouped by detected SM class, not by marketing generation:

```text
linux-amd64-sm86
linux-amd64-sm89
linux-amd64-smXX
```

`smXX` must be replaced with the detected RTX 5060/Blackwell class. Do not say that an SM 8.9 bundle supports every RTX 30xx, 40xx, or 50xx device.

A same-compute bundle is evidence only after correctness and sanity performance pass on a representative GPU with the same SM class.

## Ampere-plus

Ampere-plus is the broad desktop fallback:

```text
linux-amd64-ampere-plus
```

It is not the optimized path and may be slower than exact or same-compute artifacts. Test it separately on each desktop GPU where it is claimed. Never apply it to Jetson.

## Jetson rule

Jetson Orin Nano is exact-only:

```text
Jetson/L4T targets use exact target-local engine bundles only.
Do not publish same-compute or Ampere-plus Jetson bundles.
```

## Naming

The current packager derives names from the actual target profile:

```text
nvcr-engines-rtx3050-laptop-ubuntu2404-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-rtx4070-ubuntu2404-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-rtx5060-laptop-ubuntu2404-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-orin-nano-l4t3647-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-linux-amd64-sm86-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-linux-amd64-sm89-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-linux-amd64-smXX-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-linux-amd64-ampere-plus-dcvcrt-cvpr2025-720p.tar.gz
```

The RTX 3050 name is reserved until its profile exists; replace `smXX` with
the detected Blackwell SM before publication.

## Selection policy

The current catalog resolver ranks exact, same-compute, and Ampere-plus candidates. The current installer does not yet expose explicit `--allow-compatible` and `--allow-fallback` switches.

Until those switches exist, publication scripts and experiment runbooks must select the intended compatibility class explicitly, record it in the result row, and refuse an unplanned fallback. Adding policy flags to the installer is an open automation task.

The experiment driver never chooses a weaker class implicitly. Pass one of:

```text
--compatibility-class exact
--compatibility-class same_compute_capability
--compatibility-class ampere_plus
```

The requested value must match every selected engine manifest. It rejects
same-compute SM mismatches and rejects both compatibility classes on AArch64/
Jetson.

Compatibility runs also pass exact results from the same test target through
`--exact-baseline-jsonl`. Missing exact matches keep the package partial.
