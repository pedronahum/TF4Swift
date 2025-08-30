# TF4Swift — TensorFlow Eager in Swift (experimental)

> **Status:** early prototype. The API surface, module layout, and code generator are **subject to change** at any time. PRs welcome!

TF4Swift is a lightweight Swift wrapper around the TensorFlow **C Eager** runtime plus a code‑generated set of Swift ops and hand‑rolled automatic differentiation (AD) shims for a growing subset of math/NN primitives.

The **north star** for this project is the original **[swift-apis]** effort from the Swift for TensorFlow era: a clean, Swift‑first API layered over TensorFlow. Swift itself and the AD story have evolved since then; we’re also informed by the excellent **PassiveLogic Differentiable Swift** work and the ongoing language discussions in the Swift forums. Taken together, these let us re-think the layering and deliver a modern, pragmatic library that’s easy to build and hack on today.

- swift-apis: https://github.com/tensorflow/swift-apis  
- Differentiable Swift updates:  
  • https://forums.swift.org/t/ongoing-work-on-differentiable-swift/57780  
  • https://forums.swift.org/t/status-update-on-the-differentiable-swift-language-feature/79805  
  • Examples: https://github.com/PassiveLogic/differentiable-swift-examples


---

## Why TF4Swift?

- **Low friction**: buildable with the stock Swift toolchain + Homebrew `libtensorflow` on macOS/arm64.
- **Familiar API**: `Tensor<T>` types, `Ops` builder, and a `_Raw` namespace for non-throwing convenience.
- **AD from day one**: `relu`, `sigmoid`, and `tanh` expose reverse‑mode derivatives that compose with Swift’s `@differentiable` / pullback APIs.
- **Code generation**: Swift wrappers are emitted directly from TensorFlow’s `ops.pbtxt`, including defaulted attributes (e.g., `MatMul(transposeA: Bool = false, transposeB: Bool = false)`).
- **Opinionated but modular**: the generator writes small per‑op files grouped by domain (Math / NN / Array) under `Sources/TF4SwiftOps/Generated`.

---

## Current capabilities

- **Core**: Eager context & device listing, `Tensor` creation from scalars/arrays, shape queries, simple broadcasting utilities, diagnostics with enriched TF error messages (op name/device).
- **Ops (generated)**: `AddV2`, `Mul`, `MatMul` (with defaulted transpose flags), `Relu`, `Sigmoid`, `Tanh` (+ `ReluGrad`, `SigmoidGrad`, `TanhGrad` behind `_Raw` for AD plumbing).
- **AD**: Hand‑written `@derivative` wrappers for `relu`, `sigmoid`, `tanh` that call the TF gradient kernels and adapt to Swift’s `TangentVector`.
- **Tests**: end‑to‑end eager tests, broadcasting checks, and **golden tests** for the generator (verifying emitted signatures & builder calls). All pass on macOS/arm64 with Homebrew TF as of the latest commit.

---

## Quick start

### Prerequisites

- macOS 13+ (Ventura) with Xcode command line tools
- Swift 6.2 toolchain (package uses `// swift-tools-version: 6.2`)
- Homebrew TensorFlow C runtime:

```bash
brew install libtensorflow
# Optional: override include/lib dirs if not using Homebrew defaults
export LIBTENSORFLOW_INCLUDEDIR="/opt/homebrew/opt/libtensorflow/include"
export LIBTENSORFLOW_LIBDIR="/opt/homebrew/opt/libtensorflow/lib"
```

### Build & test

```bash
# From repo root
swift build -c release
swift test
```

### Regenerate Swift op wrappers

```bash
# Wipe old generated sources (safe: only generated files live in this folder)
rm -rf Sources/TF4SwiftOps/Generated

# Rebuild the op generator and emit wrappers
swift run -c release tf4swift-opgen

# Rebuild & test with the new sources
swift build -c release
swift test
```

> If the generator can’t find `ops.pbtxt`, ensure it’s present at `Sources/TF4SwiftOpGen/ops.pbtxt`. The tool prints the resolved path on startup.

---

## Usage examples

### Elementwise add

```swift
import TF4SwiftCore
import TF4SwiftOps

let ctx = try EagerContext()
let ops = Ops(ctx)

let a = try Tensor<Float>([1, 2, 3, 4])
let b = try Tensor<Float>([10, 20, 30, 40])
let c = try ops.add(a, b)
print(try c.shape(), c.array)  // [4] [11, 22, 33, 44]
```

### MatMul with defaults

```swift
let a = try Tensor<Float>.fromArray([1,2,3,4,5,6], shape: [2,3])
let b = try Tensor<Float>.fromArray([7,8,9,10,11,12], shape: [3,2])
let c = try ops.matmul(a, b)  // transpose flags default to false
print(try c.shape())          // [2, 2]
```

### Differentiation: `sigmoid` and `tanh`

```swift
@differentiable(reverse) func f(_ x: Tensor<Float>) -> Tensor<Float> {
    sigmoid(x) + tanh(x)
}

let x = try Tensor<Float>(0.0)
let (y, pb) = valueWithPullback(at: x, of: f)

let seed = Tensor<Float>.TangentVector(try Tensor<Float>(1.0))
let g = pb(seed)
print(y.scalar)   // ~1.5
print(g.base!)    // ~1.25 at x=0 (sigmoid’=0.25, tanh’=1.0)
```

---

## Repository layout

```
Sources/
  CTensorFlow/            # C shim for TensorFlow C & Eager (headers under include/)
  TF4SwiftCore/           # Public Swift API: EagerContext, Tensor, Ops builder, utilities
  TF4SwiftOps/            # Ops surface (depends on Core)
    _Raw.swift            # non-throwing convenience + AD helpers
    Generated/            # (codegen) organized by domain: Math/ NN/ Array/ ...
  TF4SwiftOpGen/          # Op generator tool (parsing ops.pbtxt, emitting Swift)
Tests/
  TF4SwiftCoreTests/      # Eager, broadcasting, AD, and runtime smoke tests
  TF4SwiftOpGenTests/     # Golden tests for the generator
```

---

## Roadmap

**✅ PR‑1 — Eager core & minimal ops**
- EagerContext + device listing (CPU expected)
- Tensor creation from scalars/arrays, shape helpers
- Hand‑written `Ops.add` (temporary)
- Smoke tests

**✅ PR‑2 — _Raw namespace & ReLU AD**
- `_Raw` non‑throwing conveniences
- Generated `AddV2` wrapper, diagnostics with op name/device
- `relu` + `ReluGrad` AD shims

**✅ PR‑3 — Attribute coverage + defaults + polish (OpGen)**
- Generator supports attr types: `bool`, `int`, `float`, `type`, `shape`, `string` (+ simple lists)
- `default_value` parsing surfaced as defaulted Swift params
- Consistent naming from TF arg/attr names
- File splitting by domain (Math/NN/Array)
- Golden tests for `MatMul` (transpose flags) and runtime tests
- Added `sigmoid` / `tanh` + gradient plumbing

**🔜 PR‑4 — Broader op surface**
- Arithmetic: `Sub`, `Div`, `Pow`, `Mean`, `Sum` (axes/keepDims attrs)
- Array: `Reshape`, `Concat`, `Slice`, `Gather`, `Pad`
- NN: `Conv2D` (+ attrs: strides, dilations, padding), `BiasAdd`

**🔜 PR‑5 — Higher‑level APIs**
- Simple layers (`Dense`, `Conv2D`) and an SGD/Adam training loop
- Model/Module protocol with parameter collections

**🔜 PR‑6 — UX & perf**
- Error surfaces and debug logs
- Basic benchmarking; vectorized paths where possible

**🔜 PR‑7 — Portability & CI**
- Linux support (Ubuntu + TF prebuilt)
- GitHub Actions CI: build & run tests
- Prebuilt artifacts for the opgen

**North‑star (post‑MVP)**
- Typed shapes where ergonomic, dataset utilities, export/import helpers, and a clean story for custom ops.

---

## Production‑readiness checklist

- [ ] Expand generated op coverage (see PR‑4)
- [ ] Gradients for common ops; audit correctness via numeric checks
- [ ] Memory/lifetime audits around `TFE_TensorHandle*`
- [ ] Concurrency & Sendable annotations as needed
- [ ] Linux CI + documentation examples
- [ ] API review for source stability
- [ ] Versioning + semantic tags

---

## Troubleshooting

- **Duplicate rpath warnings**: benign on macOS; we embed multiple candidates for convenience.
- **`ops.pbtxt` not found**: ensure it exists under `Sources/TF4SwiftOpGen/ops.pbtxt`. Running `swift run -c release tf4swift-opgen` should print the resolved path.
- **“value of type 'Ops' has no member 'add'”**: make sure generated sources are present under `Sources/TF4SwiftOps/Generated` and that `_Raw.swift` routes through `Ops` methods with the expected signatures.
- **AD complaints (“expression is not differentiable”)**: only call the public `relu/sigmoid/tanh` that are marked `@differentiable`. The `_Raw` helpers are intentionally not differentiated.

---

## Contributing

- Run `swift test` before submitting PRs.
- For new ops: extend the generator when possible; avoid hand‑written wrappers unless the op requires special Swift surface.
- Keep generated files in `Sources/TF4SwiftOps/Generated` only; do not edit them by hand.
- Add tests (runtime or golden) alongside new functionality.

---

## License

This project is licensed under the **Apache License 2.0**. See `LICENSE` for details. Contributions are accepted under the same license.

---

## Acknowledgements

Built on the shoulders of TensorFlow, Swift for TensorFlow alumni, and the Swift community advancing differentiable Swift.
