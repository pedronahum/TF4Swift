# TF4Swift

Swift-first, **organized** TensorFlow ops with a small core runtime and a generator that emits Swifty APIs by group (`math`, `nn`, `linalg`, …). No monolithic `_Raw` surface: each wrapper is lightweight and calls into a tiny dynamic executor.

---

## Project layout (modules)

- **CTensorFlow** — C shim over the TensorFlow C/Eager API (headers and dylibs provided by your system/ENV).
- **TF4SwiftCore** — small runtime: `EagerContext`, `Tensor`, `OpBuilder`, and the public façade **`TFExec`** which executes eager ops and forwards attributes (int/bool/string/intList). Wrappers call `TFExec` through an internal `OpRunner`.
- **TF4SwiftOps** — the public, Swifty **`Ops`** façade namespaces:
  ```swift
  let ops = Ops(ctx)
  ops.nn.relu(x)
  ops.math.tanh(x)
  ```
  Groups available: `math`, `nn`, `array`, `linalg`, `image`, `random`, `io`, `control`, `other`.
- **tf4swift-opgen** — generator that scans TF op definitions (runtime or pbtxt) + python `api_def` and emits grouped **Names** and **Wrappers** under `Sources/TF4SwiftOps/Generated/<Group>`.

---

## Install & build

### Toolchain requirement

This package currently targets Swift 6.2 toolchains. If your local `swift --version` is 6.1 (or older), you have two options:

- Install a Swift 6.2 development snapshot (recommended for contributors). On macOS, the Xcode 16/Swift 6.2 snapshot works; on Linux, install from swift.org.
- Or temporarily lower the `// swift-tools-version:` in `Package.swift` to your local version for experimentation (not recommended for CI or PRs).

Our CI uses Swift 6.2 development snapshots across macOS and Ubuntu; see the CI section below for details.

### 1) Install the TensorFlow C libraries

**macOS (Apple Silicon via Homebrew):**
```bash
brew install libtensorflow
```

You can override discovery paths with:
- `LIBTENSORFLOW_INCLUDEDIR` (default: `/opt/homebrew/opt/libtensorflow/include`)
- `LIBTENSORFLOW_LIBDIR`     (default: `/opt/homebrew/opt/libtensorflow/lib`)

Linux defaults check common `/usr/local` and `/usr` locations. You can still override using the same env vars.

### 2) Build

```bash
swift build -c release
```

---

## Generating ops (names + wrappers)

The generator reads TF op metadata from **runtime** (`TF_GetAllOpList`) or **ops.pbtxt** (bundled), and consumes the **python api_def** tree for canonical names and grouping.

**Example (with python API defs + dynamic wiring):**
```bash
.build/release/tf4swift-opgen \
  --api-def-dir "$HOME/src/tf-src/tensorflow/core/api_def/python_api" \
  --out-dir opgen-out \
  --emit-wrappers \
  --wire-dynamic \
  -v
```

What it does:
- Loads op defs (runtime → pbtxt; if runtime not available it **falls back** to the bundled `ops.pbtxt`), logs counts, and writes `op_plan.json` into `opgen-out/`.
- Emits grouped **Names** and **Wrappers** into `Sources/TF4SwiftOps/Generated/<Group>`.
- By default, wrappers are emitted for **`math`** and **`nn`**; a small allow‑list forces specific ops like `SoftmaxCrossEntropyWithLogits` under `nn`. You can change that in `WrapperEmitter`.

Optional coverage report: if present, `opgen-out/OP_COVERAGE.md` summarizes emitted unary/binary wrappers by group by scanning the generated files. If you don’t see this file, you can still gauge coverage by inspecting the generated `*_Wrappers.swift` files under `Sources/TF4SwiftOps/Generated`.

---

## Using the generated APIs

```swift
import TF4SwiftCore
import TF4SwiftOps

let ctx: EagerContext = /* your context */
let ops = Ops(ctx)

// NN
let y = ops.nn.relu(x)
// Two-outputs (loss, backprop)
let (loss, backprop) = ops.nn.softmaxCrossEntropyWithLogits(logits, labels)

// Math
let z = ops.math.tanh(x)
```

Wrappers call an internal **`OpRunner`**, which forwards to **`TFExec`** in Core. `TFExec` exposes `unary`, `binary`, plus `unary2` / `binary2` for the common two-output pattern, and accepts attribute dictionaries (`int`, `bool`, `string`, `intList`).

---

## Current generated coverage (examples)

Coverage depends on your TF version and the available `api_def` set. Typical examples you’ll see today:

- **NN**: `relu`, `selu`, `softsign`, `l2_loss`, `lrn`, and the two‑output `softmaxCrossEntropyWithLogits`.
- **Math**: `tanh`, `maximum`, `minimum`.
- **Linalg/Image/IO**: emitted as we expand the renderer rules; names indices are present and wrappers are added incrementally.

> The generator logs per‑group counts and writes exactly which functions it emitted.

---

## Configuration knobs (generator)

Edit `Sources/TF4SwiftOpGen/WrapperEmitter.swift` to adjust emission:

- `defaultGroupsToEmit`: groups that should receive wrappers by default (e.g. `["math", "nn"]`).
- `alwaysEmitOps`: graph op names that should always get wrappers (even if grouped as `other`).
- `groupOverride`: force a graph op into a specific public group (`nn`, `math`, …) for wrapper emission.
- `twoSameTypeOps`: conservative allow‑list for ops that produce **two outputs of the same element type** (until the plan adds rich `outputArgs`).

The pipeline writes a canonicalized `op_plan.json` (stable sort; pretty‑printed) to help with diffing and tests.

---

## Troubleshooting

- **dyld/loader can’t find TensorFlow libs**  
  Set:
  ```bash
  export LIBTENSORFLOW_LIBDIR=/custom/tf/lib
  export LIBTENSORFLOW_INCLUDEDIR=/custom/tf/include
  ```
  Then rebuild.

- **“Runtime load failed … Falling back to bundled ops.pbtxt”**  
  Expected if the local host doesn’t wire `TF_GetAllOpList`. The generator will keep working against the bundled pbtxt snapshot.

- **Swift tools-version mismatch (6.1 vs 6.2)**  
  If you see `package is using Swift tools version 6.2.0 but the installed version is 6.1.0`, install a 6.2 toolchain (see Toolchain requirement above) or adjust the tools-version in `Package.swift` locally for a quick test.

---

## Roadmap (completeness first)

1. **Broaden wrapper coverage** across groups (`linalg`, `image`, `io`) and surface common attributes with typed params (`axis`, `transpose/adj`, `padding`, `strides`, `dataFormat`, `dilations`).
2. **Accurate multi‑output** via richer plan data (`outputArgs`) instead of the temporary allow‑list.
3. **Grouping refinements** to migrate NN‑ish ops out of `other` as we tune the `api_def` classifier and overrides.
4. **Tests & examples** (golden `op_plan.json`, wrapper emission, eager smoke tests).
5. **(Later) Differentiation** — once API coverage is broad and stable, annotate wrappers for Swift 6 autodiff.

---

## License

TBD

---

## Continuous Integration (GitHub Actions)

The workflow `.github/workflows/ci.yml` runs on macOS and Ubuntu using Swift 6.2 development snapshots. It installs TensorFlow C (`libtensorflow` 2.19.0) via Homebrew, exports include/lib paths, and builds/tests with explicit `-I`/`-L`/`-rpath` flags. Key notes:

- Matrices: `ubuntu-22.04`, `ubuntu-24.04`, `macos-15` with Swift `6.2` (development snapshot).
- Ensures headers and dylibs are discoverable both at compile time and runtime.
- Caches SwiftPM `.build` artifacts per toolchain to speed up CI.
- Uploads the `tf4swift-opgen` executable as an artifact for convenience.

To reproduce locally:

```bash
swift --version                 # ensure 6.2 toolchain
brew install libtensorflow      # or install on Linux via your package manager
export LIBTENSORFLOW_INCLUDEDIR=$(brew --prefix libtensorflow)/include
export LIBTENSORFLOW_LIBDIR=$(brew --prefix libtensorflow)/lib
swift build -v \
  -Xcc -I"$LIBTENSORFLOW_INCLUDEDIR" \
  -Xswiftc -I"$LIBTENSORFLOW_INCLUDEDIR" \
  -Xlinker -L"$LIBTENSORFLOW_LIBDIR" \
  -Xlinker -rpath -Xlinker "$LIBTENSORFLOW_LIBDIR"
```
