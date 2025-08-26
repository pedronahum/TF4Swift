#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import CTensorFlow

public struct Ops {
    // Keep internal; no @inlinable users now
    let ctx: EagerContext

    public init(_ ctx: EagerContext) { self.ctx = ctx }

    public func build(_ name: String) -> OpBuilder { OpBuilder(ctx: ctx, name: name) }

    // Example hand-written op (useful until codegen is in place)
    public func add<T: TensorFlowScalar>(_ x: Tensor<T>, _ y: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
        let outs = try build("AddV2")
            .device(device)
            .addInput(x)
            .addInput(y)
            .attr("T", dtype: T.tfDataType)
            .execute(outputs: 1)
        return Tensor<T>(ownedHandle: outs[0])
    }
}

/// Minimal imperative builder API your generator will target.
public final class OpBuilder {
    private let st = TFStatus()
    private let op: OpaquePointer // TFE_Op*

    init(ctx: EagerContext, name: String) {
        self.op = TFE_NewOp(ctx.ptr, name, st.ptr)!
    }

    // MARK: - Configuration

    @discardableResult
    public func device(_ device: String?) -> Self {
        if let d = device, !d.isEmpty {
            TFE_OpSetDevice(op, d, st.ptr)
        }
        return self
    }

    // MARK: - Inputs

    @discardableResult
    public func addInput<T>(_ t: Tensor<T>) -> Self {
        TFE_OpAddInput(op, t.handle, st.ptr)
        return self
    }

    @discardableResult
    public func addInputList<T>(_ list: [Tensor<T>]) -> Self {
        // Eager wants an array of optional handles
        var handles: [OpaquePointer?] = list.map { Optional($0.handle) }
        handles.withUnsafeMutableBufferPointer { buf in
            TFE_OpAddInputList(op, buf.baseAddress, Int32(buf.count), st.ptr)
        }
        return self
    }

    // MARK: - Attributes (scalars)

    @discardableResult
    public func attr(_ key: String, _ v: Bool) -> Self {
        TFE_OpSetAttrBool(op, key, v ? 1 : 0)
        return self
    }

    @discardableResult
    public func attr(_ key: String, _ v: Int64) -> Self {
        TFE_OpSetAttrInt(op, key, v)
        return self
    }

    @discardableResult
    public func attr(_ key: String, _ v: Double) -> Self {
        // TF eager "float" attr is C float
        TFE_OpSetAttrFloat(op, key, Float(v))
        return self
    }

    @discardableResult
    public func attr(_ key: String, _ v: String) -> Self {
        let len = v.utf8.count
        v.withCString { cstr in
            TFE_OpSetAttrString(op, key, cstr, len)
        }
        return self
    }

    @discardableResult
    public func attr(_ key: String, dtype: TF_DataType) -> Self {
        TFE_OpSetAttrType(op, key, dtype)
        return self
    }

    /// Optional shape (nil => unknown rank; use -1 for unknown dims if needed)
    @discardableResult
    public func attr(_ key: String, shape dims: [Int64]?) -> Self {
        if let d = dims {
            d.withUnsafeBufferPointer { buf in
                TFE_OpSetAttrShape(op, key, buf.baseAddress, Int32(buf.count), st.ptr)
            }
        } else {
            // unknown rank
            TFE_OpSetAttrShape(op, key, nil, -1, st.ptr)
        }
        return self
    }

    // MARK: - Attributes (lists)

    /// String list attr. Note: TFE_OpSetAttrStringList expects `const void* const*` and **no TF_Status**.
    @discardableResult
    public func attr(_ key: String, _ values: [String]) -> Self {
        // Allocate C strings with stable lifetime for the duration of the call.
        let cstrs: [UnsafeMutablePointer<CChar>?] = values.map { strdup($0) }
        defer { cstrs.forEach { free($0) } }

        var rawPtrs: [UnsafeRawPointer?] = cstrs.map { $0.map { UnsafeRawPointer($0) } }
        var lengths: [Int] = values.map { $0.utf8.count }

        rawPtrs.withUnsafeMutableBufferPointer { pBuf in
            lengths.withUnsafeMutableBufferPointer { lBuf in
                TFE_OpSetAttrStringList(op, key,
                                        pBuf.baseAddress,
                                        lBuf.baseAddress,
                                        Int32(values.count))
            }
        }
        return self
    }

    @discardableResult
    public func attr(_ key: String, dtypes: [TF_DataType]) -> Self {
        var ts = dtypes
        ts.withUnsafeMutableBufferPointer { buf in
            TFE_OpSetAttrTypeList(op, key, buf.baseAddress, Int32(buf.count))
        }
        return self
    }

    // MARK: - Execute

    /// Execute and return `n` output handles.
    public func execute(outputs n: Int32 = 1) throws -> [OpaquePointer] {
        var nOut = n
        var outs: [OpaquePointer?] = Array(repeating: nil, count: Int(n))
        TFE_Execute(op, &outs, &nOut, st.ptr)
        try st.throwIfError()
        return outs.map { $0! } // TF guarantees filled outputs on success
    }

    deinit {
        TFE_DeleteOp(op)
    }
}
