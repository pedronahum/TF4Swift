// Sources/TF4SwiftOpGen/OpGen.swift
import Foundation

// MARK: - Models

public struct Op {
    public let name: String
    public let block: String   // full op { ... } text
}

// MARK: - pbtxt discovery

/// Try to locate the built bundle's ops.pbtxt (preferred).
func findOpsPbtxtInBuildProducts() -> URL? {
    let fm = FileManager.default
    // Work from the package root if possible, else current directory.
    let start = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
    // Common SwiftPM build folders; we'll just walk downward from CWD.
    let candidates = [
        start.appendingPathComponent(".build", isDirectory: true),
        start
    ]
    for base in candidates {
        if let it = fm.enumerator(at: base, includingPropertiesForKeys: nil) {
            for case let u as URL in it {
                if u.lastPathComponent == "ops.pbtxt",
                   u.path.contains("TF4Swift_TF4SwiftOpGen.bundle") {
                    return u
                }
            }
        }
    }
    return nil
}

/// Fallback to a repo-local checked-in pbtxt (e.g. Sources/TF4SwiftOpGen/ops.pbtxt).
func findOpsPbtxtInSources() -> URL? {
    let fm = FileManager.default
    let here = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
    let local = here.appendingPathComponent("Sources/TF4SwiftOpGen/ops.pbtxt")
    return fm.fileExists(atPath: local.path) ? local : nil
}

// MARK: - Text utilities

/// Return the smallest curly-brace block that encloses a given offset.
/// Block includes the leading "op {"/"op_def {" and its matching closing '}'.
private func enclosingCurlyBlock(around idx: String.Index, in text: String) -> Range<String.Index>? {
    // Find the *nearest* opening marker ("op {" or "op_def {") before idx.
    let markers = ["op_def {", "op {"]
    var bestOpen: Range<String.Index>? = nil

    for marker in markers {
        var search = text.startIndex
        while let r = text.range(of: marker, options: [], range: search..<idx) {
            if bestOpen == nil || r.lowerBound > bestOpen!.lowerBound {
                bestOpen = r
            }
            search = r.upperBound
        }
    }
    guard let openRange = bestOpen,
          let braceIndex = text[openRange].firstIndex(of: "{")
    else { return nil }

    // Walk forward to match braces.
    var depth = 0
    var i = braceIndex
    while i < text.endIndex {
        let ch = text[i]
        if ch == "{" { depth += 1 }
        else if ch == "}" {
            depth -= 1
            if depth == 0 {
                let end = text.index(after: i)
                return openRange.lowerBound..<end
            }
        }
        i = text.index(after: i)
    }
    return nil
}

// MARK: - find op by name

public func findOp(named name: String, in allOpsText: String) -> Op? {
    // Find 'name: "AddV2"' near the head of the block.
    guard let nameMatch = allOpsText.range(of: #"name:\s*"\#(name)""#, options: .regularExpression)
    else { return nil }

    guard let blockRange = enclosingCurlyBlock(around: nameMatch.lowerBound, in: allOpsText)
    else { return nil }

    let block = String(allOpsText[blockRange])
    return Op(name: name, block: block)
}

// MARK: - Emission helpers

private func domain(for opName: String) -> String {
    // Very small partition for now.
    switch opName {
    case "Relu", "ReluGrad", "Sigmoid", "SigmoidGrad", "Tanh", "TanhGrad":
        return "NN"
    default:
        return "Math"
    }
}

/// Emit Swift for a *known* op. We special-case the handful we support in PR-3.
public func emitSwiftForOp(_ op: Op, in allOpsText: String) throws -> (domain: String, fileName: String, contents: String, alsoEmitted: [String]) {
    switch op.name {

    // ---------- Math ----------
    case "AddV2":
        let src = """
        import TF4SwiftCore

        public extension Ops {
            func add<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
                let outs = try build("AddV2")
                    .device(device)
                    .addInput(x)
                    .addInput(y)
                    .attr("T", dtype: T.tfDataType)
                    .execute(outputs: 1)
                return Tensor<T>.fromOwnedHandle(outs[0])
            }
        }
        """
        return ("Math", "AddV2.swift", src, [])

    case "Mul":
        let src = """
        import TF4SwiftCore

        public extension Ops {
            func mul<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
                let outs = try build("Mul")
                    .device(device)
                    .addInput(x)
                    .addInput(y)
                    .attr("T", dtype: T.tfDataType)
                    .execute(outputs: 1)
                return Tensor<T>.fromOwnedHandle(outs[0])
            }
        }
        """
        return ("Math", "Mul.swift", src, [])

    case "MatMul":
        // Default transpose flags = false (matches TF defaults).
        let src = """
        import TF4SwiftCore

        public extension Ops {
            func matmul<T: TensorFlowNumeric>(
                _ a: Tensor<T>, _ b: Tensor<T>,
                transposeA: Bool = false,
                transposeB: Bool = false,
                device: String? = nil
            ) throws -> Tensor<T> {
                let outs = try build("MatMul")
                    .device(device)
                    .addInput(a)
                    .addInput(b)
                    .attr("T", dtype: T.tfDataType)
                    .attr("transpose_a", transposeA)
                    .attr("transpose_b", transposeB)
                    .execute(outputs: 1)
                return Tensor<T>.fromOwnedHandle(outs[0])
            }
        }
        """
        return ("Math", "MatMul.swift", src, [])

    // ---------- NN ----------
    case "Relu":
        // We emit both the Ops wrapper and the top-level differentiable helper here.
        let src = """
        import TF4SwiftCore
        import _Differentiation

        public extension Ops {
            func relu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
                let outs = try build("Relu")
                    .device(device)
                    .addInput(x)
                    .attr("T", dtype: T.tfDataType)
                    .execute(outputs: 1)
                return Tensor<T>.fromOwnedHandle(outs[0])
            }

            func reluGrad<T: TensorFlowFloatingPoint>(_ gradients: Tensor<T>, features x: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
                let outs = try build("ReluGrad")
                    .device(device)
                    .addInput(gradients)
                    .addInput(x)
                    .attr("T", dtype: T.tfDataType)
                    .execute(outputs: 1)
                return Tensor<T>.fromOwnedHandle(outs[0])
            }
        }

        @inlinable
        @differentiable(reverse)
        public func relu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
            _Raw.relu(x)
        }

        @inlinable
        @derivative(of: relu)
        public func _vjpRelu<T: TensorFlowFloatingPoint>(
            _ x: Tensor<T>
        ) -> (value: Tensor<T>, pullback: (Tensor<T>.TangentVector) -> Tensor<T>.TangentVector) {
            let y = _Raw.relu(x)
            return (y, { v in
                guard let seed = v.base else { return .zero }
                let g = _Raw.reluGrad(seed, x)
                return Tensor<T>.TangentVector(g)
            })
        }
        """
        return ("NN", "Relu.swift", src, ["ReluGrad.swift"])

    case "ReluGrad":
        // Emitted together with Relu.swift (avoid duplicate files).
        return ("NN", "ReluGrad.swift", "// Emitted with Relu.swift", [])

    case "Sigmoid":
        let src = """
        import TF4SwiftCore
        import _Differentiation

        public extension Ops {
            func sigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
                let outs = try build("Sigmoid")
                    .device(device)
                    .addInput(x)
                    .attr("T", dtype: T.tfDataType)
                    .execute(outputs: 1)
                return Tensor<T>.fromOwnedHandle(outs[0])
            }

            // TF signature: SigmoidGrad(y, dy)
            func sigmoidGrad<T: TensorFlowFloatingPoint>(_ y: Tensor<T>, _ dy: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
                let outs = try build("SigmoidGrad")
                    .device(device)
                    .addInput(y)
                    .addInput(dy)
                    .attr("T", dtype: T.tfDataType)
                    .execute(outputs: 1)
                return Tensor<T>.fromOwnedHandle(outs[0])
            }
        }

        @inlinable
        @differentiable(reverse)
        public func sigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
            _Raw.sigmoid(x)
        }

        @inlinable
        @derivative(of: sigmoid)
        public func _vjpSigmoid<T: TensorFlowFloatingPoint>(
            _ x: Tensor<T>
        ) -> (value: Tensor<T>, pullback: (Tensor<T>.TangentVector) -> Tensor<T>.TangentVector) {
            let y = _Raw.sigmoid(x)
            return (y, { v in
                guard let seed = v.base else { return .zero }
                let g = _Raw.sigmoidGrad(seed, y) // flip to (y, dy) inside _Raw
                return Tensor<T>.TangentVector(g)
            })
        }
        """
        return ("NN", "Sigmoid.swift", src, ["SigmoidGrad.swift"])

    case "SigmoidGrad":
        return ("NN", "SigmoidGrad.swift", "// Emitted with Sigmoid.swift", [])

    case "Tanh":
        let src = """
        import TF4SwiftCore
        import _Differentiation

        public extension Ops {
            func tanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
                let outs = try build("Tanh")
                    .device(device)
                    .addInput(x)
                    .attr("T", dtype: T.tfDataType)
                    .execute(outputs: 1)
                return Tensor<T>.fromOwnedHandle(outs[0])
            }

            // TF signature: TanhGrad(y, dy)
            func tanhGrad<T: TensorFlowFloatingPoint>(_ y: Tensor<T>, _ dy: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
                let outs = try build("TanhGrad")
                    .device(device)
                    .addInput(y)
                    .addInput(dy)
                    .attr("T", dtype: T.tfDataType)
                    .execute(outputs: 1)
                return Tensor<T>.fromOwnedHandle(outs[0])
            }
        }

        @inlinable
        @differentiable(reverse)
        public func tanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
            _Raw.tanh(x)
        }

        @inlinable
        @derivative(of: tanh)
        public func _vjpTanh<T: TensorFlowFloatingPoint>(
            _ x: Tensor<T>
        ) -> (value: Tensor<T>, pullback: (Tensor<T>.TangentVector) -> Tensor<T>.TangentVector) {
            let y = _Raw.tanh(x)
            return (y, { v in
                guard let seed = v.base else { return .zero }
                let g = _Raw.tanhGrad(seed, y) // flip inside _Raw to (y, dy)
                return Tensor<T>.TangentVector(g)
            })
        }
        """
        return ("NN", "Tanh.swift", src, ["TanhGrad.swift"])

    case "TanhGrad":
        return ("NN", "TanhGrad.swift", "// Emitted with Tanh.swift", [])

    default:
        // Not emitted in PR-3
        struct Skip: Error {}
        throw Skip()
    }
}

// Back-compat for an older golden test that expected this symbol name.
public func emitMatMul(_ op: Op, rawSource: String) throws -> String {
    let (_, _, contents, _) = try emitSwiftForOp(op, in: rawSource)
    return contents
}
