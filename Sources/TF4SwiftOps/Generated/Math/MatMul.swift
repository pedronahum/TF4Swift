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