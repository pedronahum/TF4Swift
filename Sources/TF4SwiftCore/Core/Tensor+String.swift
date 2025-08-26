#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import CTensorFlow

// Allow String as a TF scalar (but it won't satisfy Numeric).
extension String: TensorFlowScalar {
    public static var tensorFlowDataType: TensorDataType { .init(TF_STRING) }
}

public extension Tensor where Scalar == String {
    /// Create a scalar string tensor.
    convenience init(_ value: String) throws {
        // Create TF_Tensor(TF_STRING) with one TF_TString element.
        let utf8Count = value.utf8.count
        let t: OpaquePointer? = value.withCString { cptr in
            TF4SWIFT_NewTensorStringScalar(cptr, utf8Count)
        }
        precondition(t != nil, "Failed to create TF_STRING scalar tensor")
        let st = TFStatus()
        let h = TFE_NewTensorHandle(t, st.ptr)!
        TF_DeleteTensor(t)
        try st.throwIfError()
        self.init(ownedHandle: h)
    }

    /// Create a 1-D string tensor from contiguous scalars.
    convenience init(_ scalars: [String]) throws {
        let count = Int32(scalars.count)
        var lens: [Int] = scalars.map { $0.utf8.count }

        // Allocate buffers that hold exact-length (non-NUL-terminated) UTF-8 bytes.
        let cPtrs: [UnsafePointer<CChar>?] = scalars.enumerated().map { (_, s) in
            let n = s.utf8.count
            let buf = UnsafeMutablePointer<CChar>.allocate(capacity: n)
            _ = s.withCString { cstr in
                memcpy(buf, cstr, n) // copy without the trailing NUL
            }
            return UnsafePointer(buf)
        }
        defer {
            for p in cPtrs { if let pp = p { UnsafeMutablePointer(mutating: pp).deallocate() } }
        }

        let t = cPtrs.withUnsafeBufferPointer { pBuf -> OpaquePointer? in
            lens.withUnsafeMutableBufferPointer { lBuf -> OpaquePointer? in
                // reinterpret as const char* const*
                let raw = UnsafeRawPointer(pBuf.baseAddress)
                return TF4SWIFT_NewTensorStringVector(
                    raw?.assumingMemoryBound(to: UnsafePointer<CChar>?.self),
                    lBuf.baseAddress,
                    count
                )
            }
        }

        precondition(t != nil, "Failed to create TF_STRING vector tensor")
        let st = TFStatus()
        let h = TFE_NewTensorHandle(t, st.ptr)!
        TF_DeleteTensor(t)
        try st.throwIfError()
        self.init(ownedHandle: h)
    }
}

public extension Tensor where Scalar == String {
    /// Read back a scalar string (rank-0) from a TF_STRING tensor.
    var scalar: String {
        let st = TFStatus()
        let t = TFE_TensorHandleResolve(handle, st.ptr)!
        try! st.throwIfError()
        defer { TF_DeleteTensor(t) }
        precondition(TF_NumDims(t) == 0, "Tensor<String>.scalar requires rank-0 tensor")

        // The tensor's data is a single TF_TString value.
        let tstr = TF_TensorData(t)!.assumingMemoryBound(to: TF_TString.self)

        // Size and pointer to UTF-8 bytes (not NUL-terminated).
        let len = Int(TF_StringGetSize(tstr))
        let cptr = TF_StringGetDataPointer(tstr)!     // UnsafePointer<CChar>

        // Go through a raw buffer so we can decode as [UInt8] safely.
        let raw  = UnsafeRawBufferPointer(start: UnsafeRawPointer(cptr), count: len)
        return String(decoding: raw, as: UTF8.self)
    }
}

