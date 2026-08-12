# NVCR architecture

NVCR turns a learned video-compression model into a native codec application.
The model supplies compression behaviour; NVCR supplies the runtime, GPU
execution, engine selection, bitstream handling, and delivery surfaces around
it.

## Runtime map

```mermaid
flowchart LR
    App[C++ application or nvcr CLI] --> Runtime[NVCR runtime]
    Runtime --> Sessions[Encoder and decoder sessions]
    Sessions --> Codec[DCVC-RT codec adapter<br/>I/P state and native rANS]
    Codec --> Provider[TensorRT provider<br/>CUDA execution and TensorRT plans]
    Registry[Codec and provider registry] --> Runtime
    Catalog[Public engine catalog] --> Resolver[Artifact resolver]
    Resolver --> Provider
    Provider --> GPU[NVIDIA GPU]
```

The runtime owns the application-facing session lifecycle. The DCVC-RT adapter
owns codec state and entropy coding. The TensorRT provider owns GPU execution.
The resolver selects an engine that matches the active target before the
provider starts work.

## Data flow

```mermaid
flowchart LR
    Raw[Raw YUV frames] --> Encode[Encode session]
    Encode --> Adapter[DCVC-RT adapter]
    Adapter --> Engine[TensorRT engine]
    Engine --> AccessUnit[NVAU access units]
    AccessUnit --> Decode[Decode session]
    Decode --> Adapter2[DCVC-RT adapter]
    Adapter2 --> Engine2[TensorRT engine]
    Engine2 --> Output[Reconstructed YUV frames]
```

`NVAU` is NVCR's bounded, versioned access-unit format. It carries the codec,
model, ordering, dependency, and payload information needed by the decoder; the
learned payload itself remains DCVC-RT-specific.

## Engine selection

```mermaid
flowchart LR
    Detect[Detect GPU, CUDA, and TensorRT] --> Match[Match catalog entries]
    Match --> Exact[Exact target engine]
    Match --> SameSM[Same compute-capability engine]
    Match --> Broad[Ampere-and-newer engine]
    Exact --> Install[Install selected bundle]
    SameSM --> Install
    Broad --> Install
    Install --> Run[TensorRT provider runs NVCR]
```

NVCR prefers an exact engine. Desktop targets can use a published matching
compute-capability or broader compatibility bundle when available; Jetson uses
exact-target engines.

## Repository map

```mermaid
flowchart TB
    Repo[NVCR repository]
    Repo --> Public[include/nvcr<br/>Public C++ API]
    Repo --> RuntimeSrc[src<br/>Runtime, codec, provider, bitstream]
    Repo --> CLI[cli<br/>The nvcr command]
    Repo --> Tools[scripts<br/>Install, artifacts, validation]
    Repo --> Tests[tests<br/>Contracts and regression tests]
    Repo --> Docs[docs<br/>Guides and technical reference]
    Repo --> Results[results<br/>Target execution results]
    Public --> RuntimeSrc
    RuntimeSrc --> CLI
    RuntimeSrc --> Tests
    Tools --> Results
```

For the public C++ surface, see the [API reference](reference.md). For the
DCVC-RT implementation, see [DCVC-RT integration](dcvcrt-integration.md). For
the access-unit format, see [Bitstreams and access units](bitstream.md).
