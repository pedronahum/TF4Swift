#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import CTensorFlow
import Foundation

public struct Ops {
    // Keep internal; no @inlinable users now
    let ctx: EagerContext

    public init(_ ctx: EagerContext) { self.ctx = ctx }

    public func build(_ name: String) -> OpBuilder { OpBuilder(ctx: ctx, name: name) }

}

#if DEBUG && DEBUG_BROADCAST_CHECKS
extension Ops {
    /// DEBUG-only helper: add with host-side broadcasting preflight.
    /// Kept `internal` so there is no collision with TF4SwiftOps' public `add`.
    @usableFromInline
    internal func _addWithPreflight<T: TensorFlowNumeric>(
        _ x: Tensor<T>, _ y: Tensor<T>, device: String? = nil
    ) throws -> Tensor<T> {
        // Host-side broadcast check for nicer debug-time errors
        let sx = try x.shape()
        let sy = try y.shape()
        _ = try broadcastedShape(sx, sy)  // throws BroadcastError on mismatch

        let outs = try build("AddV2")
            .device(device)
            .addInput(x)
            .addInput(y)
            .attr("T", dtype: T.tfDataType)
            .execute(outputs: 1)
        return Tensor<T>.fromOwnedHandle(outs[0])
    }
}
#endif


/// Minimal imperative builder API your generator will target.
public final class OpBuilder {
    private let st = TFStatus()
    @usableFromInline internal let op: OpaquePointer // TFE_Op*

    @usableFromInline internal init(ctx: EagerContext, name: String) {
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

    /// Execute the prepared op and return a single Tensor output.
    /// Adjust the Tensor initializer below if your type uses a different label.
    public func runOne<T>() -> Tensor<T> {
        let st = TF_NewStatus()
        defer { TF_DeleteStatus(st) }

        var numOut: Int32 = 1
        var outHandle: OpaquePointer? = nil

        TFE_Execute(self.op, &outHandle, &numOut, st)

        // 0 == OK
        // 0 == OK
        let code = TF_GetCode(st)
        if code.rawValue != 0 {
            let msg = String(cString: TF_Message(st))
            fatalError("[TF4Swift] TFE_Execute failed: \(msg)")
        }


        guard let h = outHandle else {
            fatalError("[TF4Swift] TFE_Execute returned no output handle")
        }

        // IMPORTANT: match your Tensor initializer label.
        // Your earlier error shows `ownedHandle:` is the correct one.
        return Tensor<T>(ownedHandle: h)
    }


    // MARK: - Execute
    public func execute(outputs count: Int) throws -> [OpaquePointer] {
    var numOuts: Int32 = Int32(count)
    var outs = [OpaquePointer?](repeating: nil, count: count)

    // Execute the op.
    TFE_Execute(op, &outs, &numOuts, st.ptr)

    // If TF reported an error, decorate it with op/device info.
    do {
        try st.throwIfError()
    } catch TensorFlowError.status(let code, let tfMsg) {
        // Best-effort device and op name for diagnostics.
        var deviceStr = "unspecified"
        if let devCStr = TFE_OpGetDevice(op, st.ptr) {
            deviceStr = String(cString: devCStr)
        }
        let opName = String(cString: TFE_OpGetName(op, st.ptr))

        let msg = "Op \(opName) on device \(deviceStr) failed: \(tfMsg)"
        throw TensorFlowError.status(code: code, message: msg)
    }

    // Success: return realized outputs (trim to numOuts).
    return outs.prefix(Int(numOuts)).map { $0! }
}

    // MARK: - Attribute setters (minimal)

    /// Set an integer attribute on the current op (e.g., "axis").
    public func setAttrInt(_ name: String, _ value: Int64) {
        name.withCString { cname in
            TFE_OpSetAttrInt(self.op, cname, value)
        }
    }

    /// Execute the prepared op and return two Tensor outputs.
    /// Adjust the Tensor initializer below if your type uses a different label.
    public func runTwo<T>() -> (Tensor<T>, Tensor<T>) {
    let st = TF_NewStatus()
    defer { TF_DeleteStatus(st) }

    var numOut: Int32 = 2
    var out0: OpaquePointer? = nil
    var out1: OpaquePointer? = nil

    // Collect pointers into a tiny stack buffer to pass into TFE_Execute
    withUnsafeMutablePointer(to: &out0) { p0 in
        withUnsafeMutablePointer(to: &out1) { p1 in
        p0.withMemoryRebound(to: OpaquePointer?.self, capacity: 2) { outArray in
            // outArray[0] points at out0; outArray[1] must point at out1
            (outArray + 1).initialize(to: p1.pointee)
            TFE_Execute(self.op, outArray, &numOut, st)
        }
        }
    }

    let code = TF_GetCode(st)
    if code.rawValue != 0 {
        let msg = String(cString: TF_Message(st))
        fatalError("[TF4Swift] TFE_Execute(2 outputs) failed: \(msg)")
    }

    guard let h0 = out0, let h1 = out1 else {
        fatalError("[TF4Swift] TFE_Execute returned missing output handles")
    }

    // IMPORTANT: match your Tensor initializer label.
    // You previously compiled with `ownedHandle:`—keep that here too.
    let t0: Tensor<T> = Tensor<T>(ownedHandle: h0)
    let t1: Tensor<T> = Tensor<T>(ownedHandle: h1)
    return (t0, t1)
    }


    deinit {
        TFE_DeleteOp(op)
    }
}




