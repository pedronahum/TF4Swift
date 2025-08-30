import CTensorFlow

public extension Tensor {
    /// Returns the shape (per-dimension sizes) for this tensor.
    func shape() throws -> [Int64] {
        let st = TFStatus()
        guard let t = TFE_TensorHandleResolve(handle, st.ptr) else {
            try st.throwIfError()
            return []
        }
        defer { TF_DeleteTensor(t) }
        try st.throwIfError()

        let nd = TF_NumDims(t)
        var dims: [Int64] = []
        dims.reserveCapacity(Int(nd))
        for i in 0..<nd {
            dims.append(Int64(TF_Dim(t, i)))
        }
        return dims
    }
}
