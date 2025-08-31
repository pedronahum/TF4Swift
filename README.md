# TF4Swift — TensorFlow for Swift (experimental)

> **Status:** early research prototype. Expect sharp edges. The API surface *will* change as we iterate.

TF4Swift is a lightweight, Swifty wrapper over the TensorFlow **C & Eager** APIs that lets you build and run
TensorFlow ops directly from Swift. The project’s north star is the original
Swift for TensorFlow `swift-apis`, but implemented with **modern Swift** (Swift 6 toolchain, SwiftPM) and
informed by the **Differentiable Swift** work that has continued in the community. See:
- https://github.com/tensorflow/swift-apis
- https://forums.swift.org/t/ongoing-work-on-differentiable-swift/57780
- https://forums.swift.org/t/status-update-on-the-differentiable-swift-language-feature/79805
- https://github.com/PassiveLogic/differentiable-swift-examples

---

## Why Swift + TensorFlow C?

- **A Swifty API over mature kernels.** We reuse TensorFlow’s highly–optimized C/C++ kernels via the stable C API.
- **Safety & ergonomics.** Strong typing, `throws`, and value semantics for common paths.
- **Differentiable Swift.** Use Swift’s reverse–mode AD to prototype ML code with familiar Swift syntax.
- **Zero Python runtime dependency.** Pure Swift package; works in command line apps and server processes.

---

## What’s here today

**Core runtime**
- Minimal `EagerContext` wrapper with device enumeration and error propagation.
- `Tensor<Scalar>` value type with convenience initializers (scalars, 1D/2D/3D), shape checks, and row‑major `scalars` accessors.
- `OpBuilder` with helpful diagnostics (op name, device, and TF message on failure).
- Broadcasting preflight checks behind a debug flag.

**Ops**
- Elementwise: `add`, `mul`.
- Matrix: `matmul` with defaulted transpose flags.
- NN activations: `relu`, `sigmoid`, `tanh`.

**AD (reverse mode)**
- Correct VJP definitions for `relu` (`ReluGrad`), `sigmoid` (`SigmoidGrad`), and `tanh` (`TanhGrad`), wired to Swift’s tangent system.

**Op Generator (OpGen)**
- Reads TensorFlow’s `ops.pbtxt`, parses input/attr/type metadata, and emits Swift wrappers into `Sources/TF4SwiftOps/Generated/<Domain>/…`.
- Attribute coverage: `bool`, `int`, `float`, `type`, `shape`, `string` (+ simple lists), including default values in signatures.
- File splitting by domain (Math / NN / Array) to keep sources small.

**Tests**
- Eager smoke tests for scalars/arrays/devices, broadcasting utilities, diagnostics.
- Golden tests for the generator (e.g. MatMul has defaulted flags; emitted text calls through the builder).

---

## Roadmap (updated after external LLM reviews)

Based on feedback from Grok and Gemini, we’ve adjusted scope and ordering to prioritize **stability, testability, and
CI portability** before broad op coverage.

### Milestone PRs

- **PR‑0 — Bootstrap** ✓  
  CTensorFlow shim, `EagerContext`, `Tensor` shell, basic AddV2 wrapper, tests.
- **PR‑1 — Tensors & Creation** ✓  
  Dense initializers, shape inference, ragged/element‑type checks, `scalars` access.
- **PR‑2 — Builder & Diagnostics** ✓  
  `OpBuilder`, enriched TF error messages, debug broadcasting checks.
- **PR‑3 — OpGen attributes + defaults** ✓  
  Emit `bool/int/float/type/shape/string` attrs with defaults, domain split; runtime tests for MatMul flags.
- **PR‑4 — Unary activations w/ custom VJPs** ✓  
  `relu/sigmoid/tanh` + `ReluGrad/SigmoidGrad/TanhGrad`; AD unit tests.
- **PR‑5 — Hardening & CI (in progress)**  
  - Thread‑safe/global‐context policy + docs
  - Swift 6.2+ build on macOS & Linux (CI matrix)
  - Bench smoke tests; avoid regressions
  - Clear versioning policy for breaking changes
- **PR‑6 — Coverage & high‑level utilities (planned)**  
  - More math/array/reduction ops via OpGen  
  - Minimal `Module` + optimizers (SGD/Adam) & a tiny training loop  
  - Small `Dataset` helpers for in‑memory arrays

### Cross‑cutting work items

- **Binary compatibility**: keep to TF C/Eager APIs; document minimum libtensorflow version.  
- **Threading & performance**: document context use; micro‑benchmarks; avoid unnecessary copies.
- **Diagnostics**: consistent messages with op name and device; guidance to fix common errors.
- **Docs & samples**: runnable code snippets; cookbook for common layers.
- **Contributor guide**: coding style, test strategy, review checklist.

---

## Getting started

### Prerequisites

- macOS 13+ (Swift 6 toolchain). Linux support planned.  
- TensorFlow **C** library (`libtensorflow`). Homebrew on Apple Silicon:  
  `brew install libtensorflow`

SwiftPM is configured to look for headers and libraries under Homebrew defaults and respects environment overrides:

- `LIBTENSORFLOW_INCLUDEDIR` – fallback: `/opt/homebrew/opt/libtensorflow/include`  
- `LIBTENSORFLOW_LIBDIR` – fallback: `/opt/homebrew/opt/libtensorflow/lib`

Run:
```bash
swift build
swift test
```

If you installed TensorFlow outside Homebrew, set the env vars before building:
```bash
export LIBTENSORFLOW_INCLUDEDIR=/path/to/include
export LIBTENSORFLOW_LIBDIR=/path/to/lib
swift build
```

### Generate (or re‑generate) ops

The generator reads `ops.pbtxt` and writes Swift wrappers under `Sources/TF4SwiftOps/Generated/…`.

```bash
# Optional: remove previously generated files
rm -rf Sources/TF4SwiftOps/Generated

# Rebuild the generator
swift run -c release tf4swift-opgen

# Build & test the package
swift build -c release
swift test
```

**Tip:** If you see errors like “`value of type 'Ops' has no member …`” it usually means the corresponding generated
file is missing. Remove `Sources/TF4SwiftOps/Generated` and re‑run the generator.

---

## Usage

```swift
import TF4SwiftCore
import TF4SwiftOps
import _Differentiation

let ctx = try EagerContext()
let ops = Ops(ctx)

// Tensors
let a = try Tensor<Float>([1, 2, 3, 4])
let b = try Tensor<Float>([10, 20, 30, 40])

// Elementwise
let c = try ops.add(a, b)           // [11, 22, 33, 44]

// MatMul (with defaulted transpose flags)
let A = try Tensor<Float>.fromArray([1,2,3,4,5,6], shape: [2,3])
let B = try Tensor<Float>.fromArray([7,8,9,10,11,12], shape: [3,2])
let C = try ops.matmul(A, B)        // 2x2

// AD examples
@differentiable(reverse)
func relu(_ x: Tensor<Float>) -> Tensor<Float> { _Raw.relu(x) }

let x = try Tensor<Float>([-1, 0, 2])
let (y, pb) = valueWithPullback(at: x, of: relu)
let g = pb(.init(try Tensor<Float>([1, 1, 1]))).base  // ∂relu/∂x ⋅ seed
```

---

## Troubleshooting

- **Duplicate `-rpath` warnings.** Benign on macOS; we embed multiple rpaths so `libtensorflow` can be found at runtime.
- **`Ops` has no member `xyz`.** Remove `Sources/TF4SwiftOps/Generated` and re‑run `tf4swift-opgen`.
- **`op not found: AddV2` in OpGen.** Ensure the `ops.pbtxt` the tool reads matches your TF install; check the path printed by the tool and replace if needed.
- **`dyld: Library not loaded: libtensorflow`.** Verify `LIBTENSORFLOW_LIBDIR` or install via Homebrew; the build embeds an rpath to the lib directory.

---

## Contributing

This is an experiment. Contributions are welcome, but **APIs are unstable**.  
Please open an issue for design topics, especially around AD, context management,
and code‑generation conventions.

---

## License

Licensed under the **Apache License 2.0**. See `LICENSE` in this repo.

---

## Acknowledgements

- The Swift for TensorFlow project (`swift-apis`) for inspiring the API shape and documentation examples.
- The community work on Differentiable Swift for showing viable paths forward with modern toolchains.
- TensorFlow maintainers for the stable C API we build upon.
