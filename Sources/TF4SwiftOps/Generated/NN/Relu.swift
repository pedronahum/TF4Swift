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