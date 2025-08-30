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