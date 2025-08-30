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