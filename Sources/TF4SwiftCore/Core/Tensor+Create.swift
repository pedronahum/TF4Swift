// Sources/TF4SwiftCore/Core/Tensor+Create.swift
import CTensorFlow

public extension Tensor where Scalar: TensorFlowScalar {
    /// Build a dense tensor from a flat Swift array with an explicit shape.
    /// `shape.product == scalars.count` must hold.
    static func fromArray(_ scalars: [Scalar], shape: [Int64]) throws -> Tensor<Scalar> {
        precondition(shape.reduce(1, *) == scalars.count,
                     "shape \(shape) does not match element count \(scalars.count)")

        let elementBytes = MemoryLayout<Scalar>.stride
        let totalBytes   = elementBytes * scalars.count

        // Allocate TF_Tensor with the provided shape and copy bytes.
        let tfTensor: OpaquePointer = shape.withUnsafeBufferPointer { dims -> OpaquePointer in
            let t = TF_AllocateTensor(Scalar.tfDataType, dims.baseAddress, Int32(dims.count), totalBytes)!
            _ = scalars.withUnsafeBytes { raw in
                memcpy(TF_TensorData(t), raw.baseAddress!, raw.count)
            }
            return t
        }

        // Wrap in an eager handle.
        let st = TFStatus()
        let h  = TFE_NewTensorHandle(tfTensor, st.ptr)!
        TF_DeleteTensor(tfTensor)
        try st.throwIfError()

        return Tensor<Scalar>.fromOwnedHandle(h)
    }

    /// 2‑D convenience initializer for rectangular matrices.
    convenience init(_ values2D: [[Scalar]]) throws {
        let rows = values2D.count
        precondition(rows > 0, "empty outer array")
        let cols = values2D[0].count
        precondition(values2D.allSatisfy { $0.count == cols }, "ragged 2‑D array")

        var flat: [Scalar] = []
        flat.reserveCapacity(rows * cols)
        for r in values2D { flat.append(contentsOf: r) }

        let elementBytes = MemoryLayout<Scalar>.stride
        let totalBytes   = elementBytes * flat.count
        let dims: [Int64] = [Int64(rows), Int64(cols)]

        let tfTensor: OpaquePointer = dims.withUnsafeBufferPointer { buf -> OpaquePointer in
            let t = TF_AllocateTensor(Scalar.tfDataType, buf.baseAddress, Int32(buf.count), totalBytes)!
            _ = flat.withUnsafeBytes { raw in
                memcpy(TF_TensorData(t), raw.baseAddress!, raw.count)
            }
            return t
        }

        let st = TFStatus()
        let h  = TFE_NewTensorHandle(tfTensor, st.ptr)!
        TF_DeleteTensor(tfTensor)
        try st.throwIfError()

        self.init(ownedHandle: h)
    }
}
