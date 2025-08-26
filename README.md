# TF4Swift — TensorFlow for Swift (experimental)

> **Status:** 🚧 Experimental & subject to change.  
> This project explores a modern, type-safe Swift interface to TensorFlow’s C & Eager APIs, plus early automatic differentiation (AD) integration in Swift 6.2 development toolchains.

---

## Why TensorFlow for Swift?

Swift gives us:
- **Type safety & ergonomics** — expressive generics and protocol constraints (e.g., `TensorFlowScalar`, `TensorFlowNumeric`, `TensorFlowFloatingPoint`) make invalid combinations harder to write and easier to catch at compile time.
- **Great tooling** — SwiftPM, the new `swift-testing` library, first-class async/actors (future direction for device/runtime management).
- **AD in the language toolchain** — Swift 6.2 dev includes reverse-mode differentiation attributes. We can declare `@differentiable(reverse)` entry points and provide custom VJPs that call TensorFlow’s gradient kernels.

TensorFlow gives us:
- **Battle-tested kernels and devices** — you call into the existing C/Eager runtime and ops (CPU/GPU/accelerators where available).
- **Broad operator coverage** — we can generate most wrappers directly from TensorFlow’s `ops.pbtxt`.

Together, this aims to be a practical “Swift-first” experience on top of TensorFlow while preserving Swift’s design principles.

---

## What works today (MVP)

- A small **C shim** (`CTensorFlow`) over the TensorFlow C & Eager API (plus string helpers), linked against your system `libtensorflow`.
- **Core Swift API** (`TF4SwiftCore`): eager `Tensor`, `EagerContext`, a fluent `Ops` builder, and dtype protocols.
- **Op generator** (`tf4swift-opgen`) that reads `tensorflow/core/ops/ops.pbtxt` and emits Swift wrappers. We currently generate:
  - `AddV2` (numeric), `Relu` (floating-point), `ReluGrad` + Swift AD glue for `relu(_:)`.
- **Swift AD integration**: floating `Tensor<T>` conforms to `Differentiable`; we expose a differentiable `relu` with a pullback implemented by TensorFlow’s `ReluGrad`.

> Manifest at a glance: products `TF4SwiftCore`, `TF4SwiftOps`, and tool `tf4swift-opgen`; the manifest wires header search paths and rpaths for `libtensorflow`. See `Package.swift` for details. fileciteturn0file0 fileciteturn0file1

---

## Repository layout

```
TF4Swift/
├─ Sources/
│  ├─ CTensorFlow/           # C headers & shims over TF C/Eager + tiny helpers
│  ├─ TF4SwiftCore/          # Public Swift API: Tensor, Ops, EagerContext, dtype protocols
│  ├─ TF4SwiftOps/           # Generated high-level op wrappers (one file per op)
│  └─ TF4SwiftOpGen/         # Code generator tool (parses ops.pbtxt, emits Swift)
└─ Tests/
   └─ TF4SwiftCoreTests/     # swift-testing based tests (eager, ops, AD)
```

---

## Getting started

### Prerequisites

- macOS 13+ (Apple Silicon recommended)
- Swift 6.2 **development** toolchain (or newer)
- TensorFlow C library installed (e.g., Homebrew `libtensorflow`)
- Environment variables telling the build where headers & libs live:

```bash
export LIBTENSORFLOW_INCLUDEDIR=/opt/homebrew/opt/libtensorflow/include
export LIBTENSORFLOW_LIBDIR=/opt/homebrew/opt/libtensorflow/lib
```

### Build

```bash
swift build -c release
```

### Run the op generator

Place `ops.pbtxt` in `Sources/TF4SwiftOpGen/` (or set `TF_OPS_PBTXT`), then:

```bash
swift run -c release tf4swift-opgen
# Generates Swift files under Sources/TF4SwiftOps/Generated/
```

### Test

```bash
swift test -v
```

> If your tests import `TF4SwiftOps`, make sure the test target depends on that product in `Package.swift`.

---

## Usage snippets

### Eager add & ReLU

```swift
import TF4SwiftCore
import TF4SwiftOps

let ctx = try EagerContext()
let ops = Ops(ctx)

let a = try Tensor<Float>(3.0)
let b = try Tensor<Float>(4.0)
let c = try ops.add(a, b)         // 7.0

let x = try Tensor<Float>([-1, 0, 2])
let y = try ops.relu(x)           // [0, 0, 2]
```

### AD with a custom VJP backed by TF grads (Swift 6.2 dev)

```swift
import _Differentiation
import TF4SwiftCore
import TF4SwiftOps

let x = try Tensor<Float>([-1, 0, 2])
let (y, pb) = valueWithPullback(at: x, of: relu)
let seed = try Tensor<Float>([1, 1, 1]) // upstream cotangent
let grad = pb(.init(seed)).base!        // [0, 0, 1]
```

---

## Roadmap & milestones

We’re starting small and iterating. Expect breaking changes while we find the right shapes and protocols.

### Phase 0 – Housekeeping (quick wins)
- Ensure tests that import `TF4SwiftOps` declare a dependency on it.
- Lock in AD attribute & pullback conventions.

### Phase 1 – Core API hardening
- Tensor creation (ND initializers, zeros/ones/like, random, range/linspace).
- Device/runtime ergonomics (`EagerContext` lifecycle, device lists, per-op placement).
- Error surface & diagnostics (clear messages on dtype/shape mismatches).
- Memory safety audit (correct deletion of handles in all paths).
- Protocol lattice finalization (`TensorFlowScalar/Numeric/FloatingPoint/Integer/Index`).

**Acceptance:** shape/dtype tests, broadcast checks, device smoke tests.

### Phase 2 – Production-ready op generation
- Robust parsing of `ops.pbtxt` (`type_attr`, lists, shapes, allowed dtypes).
- Protocol binding from attr constraints (float-only, integer-only, numeric, scalar).
- Input lists, multi-output ops (tuple returns with labels).
- Attribute setters (int/float/bool/string/dtype/shape).
- Grad pair detection + optional AD glue for a **whitelist** (ReLU/Sigmoid/Tanh, etc.).
- File-per-op generation under `TF4SwiftOps/Generated/`.

**Acceptance:** autogen coverage for `AddV2`, `Relu`, `ReluGrad`, `Mul`, `Sub`, `Neg`, `MatMul`, `Concat`, `Reshape`, `TopKV2`, `Pack`, `Unpack`; unit tests.

### Phase 3 – Differentiation beyond unary activations
- `Tensor<T>` AD: refine zero tangents (materialize `zerosLike` when shapes known).
- AD rules for binary ops (`add/sub/mul/div`) and reductions (`sum/mean/max`).
- Broadcasting-aware VJPs; finite-difference checks in tests.

**Acceptance:** numeric vs. AD agreement on random small tensors.

### Phase 4 – DX, CI, docs
- `swift-testing` property tests; end-to-end generator/build/test pipeline.
- `swift-format` for both handwritten and generated sources.
- CI matrix (macOS 13+/15, Swift 6.2 dev) building core → generator → ops → tests.
- Examples: eager basics, a tiny training loop once we add more grads.

### Nice-to-have tracks (in parallel)
- String tensors (`TF_TString`-backed `StringTensor` type).
- SavedModel/graph mode exploration.
- Performance (context reuse, op caching, fused kernels).
- Device utilities (explicit CPU/GPU selection, soft-placement).

---

## Next steps

1. **Expand generator** to handle input lists, multi-outputs, and attr binding from `allowed_values`.
2. **Add array initializers** (shape inference) and broadcasting tests.
3. **Add VJPs** for `AddV2` and `Mul`; compare to finite differences in tests.
4. **Wire CI** that runs: build → opgen → build ops → test.

---

## License

**Apache License, Version 2.0** — recommended for this project.  
You’ll find the license text in `LICENSE` at the repo root. Contributions are assumed to be licensed under Apache-2.0 unless stated otherwise.

> Add a `LICENSE` file with the standard Apache 2.0 text and a `NOTICE` file if needed (attribution statements). A typical copyright header:
>
> ```
> Copyright 2025 The TF4Swift Authors
>
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.
> ```

---

## Contributing

This is an early, fast-moving experiment. Issues, design notes, and small PRs are welcome. Please open an issue before larger refactors so we can agree on direction (especially around AD semantics and codegen).

---


