// Sources/TF4SwiftOps/_Raw.swift
import TF4SwiftCore

public enum _Raw {}

public extension _Raw {
    // Process-global eager context for convenience wrappers.
    static let defaultContext: EagerContext = {
        try! EagerContext()
    }()

    // -------- Math --------
    static func add<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>, device: String? = nil) -> Tensor<T> {
        try! Ops(defaultContext).add(x, y, device: device)
    }

    static func mul<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>, device: String? = nil) -> Tensor<T> {
        try! Ops(defaultContext).mul(x, y, device: device)
    }

    static func matmul<T: TensorFlowNumeric>(
        _ a: Tensor<T>, _ b: Tensor<T>,
        transposeA: Bool = false,
        transposeB: Bool = false,
        device: String? = nil
    ) -> Tensor<T> {
        try! Ops(defaultContext).matmul(a, b, transposeA: transposeA, transposeB: transposeB, device: device)
    }
}

public extension _Raw {
    // -------- NN --------
    static func relu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, device: String? = nil) -> Tensor<T> {
        try! Ops(defaultContext).relu(x, device: device)
    }

    // Accept (seed, x); TF op is ReluGrad(gradients, features).
    static func reluGrad<T: TensorFlowFloatingPoint>(_ seed: Tensor<T>, _ x: Tensor<T>, device: String? = nil) -> Tensor<T> {
        try! Ops(defaultContext).reluGrad(seed, features: x, device: device)
    }

    static func sigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, device: String? = nil) -> Tensor<T> {
        try! Ops(defaultContext).sigmoid(x, device: device)
    }

    // Accept (seed, y) to match pullback <seed>; TF op is SigmoidGrad(y, dy).
    static func sigmoidGrad<T: TensorFlowFloatingPoint>(_ seed: Tensor<T>, _ y: Tensor<T>, device: String? = nil) -> Tensor<T> {
        try! Ops(defaultContext).sigmoidGrad(y, seed, device: device)
    }

    static func tanh<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, device: String? = nil) -> Tensor<T> {
        try! Ops(defaultContext).tanh(x, device: device)
    }

    // Accept (seed, y) to match pullback; TF op is TanhGrad(y, dy).
    static func tanhGrad<T: TensorFlowFloatingPoint>(_ seed: Tensor<T>, _ y: Tensor<T>, device: String? = nil) -> Tensor<T> {
        try! Ops(defaultContext).tanhGrad(y, seed, device: device)
    }
}
