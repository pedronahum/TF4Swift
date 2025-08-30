import CTensorFlow

import CTensorFlow

public extension Tensor where Scalar: TensorFlowScalar {
    /// Flattened host copy of all elements, in row‑major order, for tensors of any rank.
    ///
    /// - Note: This property is intended for debugging/tests. It resolves the
    ///   eager handle to a host `TF_Tensor` and copies bytes into a Swift array.
    var scalars: [Scalar] {
        let st = TFStatus()
        guard let t = TFE_TensorHandleResolve(handle, st.ptr) else {
            // If resolve failed, status will carry the reason.
            let msg = String(cString: TF_Message(st.ptr))
            fatalError("Failed to resolve tensor handle: \(msg)")
        }
        defer { TF_DeleteTensor(t) }

        let byteCount = Int(TF_TensorByteSize(t))
        let stride    = MemoryLayout<Scalar>.stride
        precondition(stride > 0, "Invalid element size")
        let count     = byteCount / stride

        return [Scalar](unsafeUninitializedCapacity: count) { buf, initialized in
            memcpy(buf.baseAddress!, TF_TensorData(t), byteCount)
            initialized = count
        }
    }
}
