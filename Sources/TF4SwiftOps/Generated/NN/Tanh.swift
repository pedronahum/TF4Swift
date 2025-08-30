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