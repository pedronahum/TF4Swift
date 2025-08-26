#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import CTensorFlow

/// Swift wrapper for an eager tensor handle (TFE_TensorHandle*).
public final class Tensor<Scalar: TensorFlowScalar> {
    public var handle: OpaquePointer  // TFE_TensorHandle*

    // Designated: take ownership of a TFE_TensorHandle*
    init(ownedHandle h: OpaquePointer) { self.handle = h }

    deinit { TFE_DeleteTensorHandle(handle) }

    // MARK: - Create from a scalar

    /// Initialize from a single scalar value.
    public convenience init(_ value: Scalar) throws {
        var v = value
        let elemSize = MemoryLayout<Scalar>.stride
        // Build a TF_Tensor for a scalar (0 dims), then wrap into an eager handle.
        let t = withUnsafeBytes(of: &v) { raw -> OpaquePointer in
            let ptr = TF4SWIFT_NewTensorScalar(Scalar.tfDataType, raw.baseAddress, elemSize)!
            return ptr
        }
        let st = TFStatus()
        let h = TFE_NewTensorHandle(t, st.ptr)!
        TF_DeleteTensor(t)
        try st.throwIfError()
        self.init(ownedHandle: h)
    }

    // MARK: - Create from a 1-D array

    /// Initialize from a 1-D array `[Scalar]`.
    public convenience init(_ values: [Scalar]) throws {
        var dims: [Int64] = [Int64(values.count)]
        let byteCount = values.count * MemoryLayout<Scalar>.stride
        let t = TF_AllocateTensor(Scalar.tfDataType, &dims, 1, byteCount)!
        // Copy bytes into TF_Tensor storage.
        values.withUnsafeBytes { raw in
            let dst = TF_TensorData(t)!
            dst.copyMemory(from: raw.baseAddress!, byteCount: byteCount)
        }
        let st = TFStatus()
        let h = TFE_NewTensorHandle(t, st.ptr)!
        TF_DeleteTensor(t)
        try st.throwIfError()
        self.init(ownedHandle: h)
    }

    /// Convenience: allow initializing a Float/Double tensor from an `[Int]` (or any BinaryInteger).
    public convenience init<I: BinaryInteger>(_ ints: [I]) throws where Scalar: BinaryFloatingPoint {
        try self.init(ints.map { Scalar($0) })
    }

    /// Convenience: allow initializing an Int32/Int64 tensor from an `[Int]` (or any BinaryInteger).
    public convenience init<I: BinaryInteger>(_ ints: [I]) throws where Scalar: FixedWidthInteger & SignedInteger {
        try self.init(ints.map { Scalar(truncatingIfNeeded: $0) })
    }

    // MARK: - Read back

    /// Fetch the scalar value (tensor must be rank-0).
    public var scalar: Scalar {
        let st = TFStatus()
        let t = TFE_TensorHandleResolve(handle, st.ptr)!
        // If this throws, crash with a clear error; tests rely on correct shapes.
        try! st.throwIfError()
        defer { TF_DeleteTensor(t) }
        precondition(TF_NumDims(t) == 0, "Tensor.scalar requires rank-0 tensor")
        let ptr = TF_TensorData(t)!.assumingMemoryBound(to: Scalar.self)
        return ptr.pointee
    }

    /// Read back a 1-D array.
    public var array: [Scalar] {
        let st = TFStatus()
        let t = TFE_TensorHandleResolve(handle, st.ptr)!
        try! st.throwIfError()
        defer { TF_DeleteTensor(t) }
        precondition(TF_NumDims(t) == 1, "Tensor.array only supports rank-1 tensors")
        let n = Int(TF_Dim(t, 0))
        let ptr = TF_TensorData(t)!.assumingMemoryBound(to: Scalar.self)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    /// Public factory for other modules (e.g., TF4SwiftOps) to adopt owned handles.
    public static func fromOwnedHandle(_ h: OpaquePointer) -> Tensor<Scalar> {
        return Tensor<Scalar>(ownedHandle: h) // designated initializer is internal; this factory is public
    }

}

#if canImport(_Differentiation)
import _Differentiation
import CTensorFlow

extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// Convenience: convert a tensor to its tangent vector (used as AD seed).
    @inlinable public var tangentVector: TangentVector { .init(self) }
}

/// Tangent for `Tensor<Scalar>` where `Scalar` is floating-point.
/// We use `nil` as a "zero" sentinel to avoid shape bookkeeping.
@frozen
public struct _TensorTangent<Scalar: TensorFlowFloatingPoint>: AdditiveArithmetic, Differentiable, Equatable {
    public var base: Tensor<Scalar>?   // nil == zero
    @inlinable public init(_ base: Tensor<Scalar>? = nil) { self.base = base }

    // MARK: AdditiveArithmetic
    @inlinable public static var zero: Self { .init(nil) }

    @inlinable
    public static func + (lhs: Self, rhs: Self) -> Self {
        switch (lhs.base, rhs.base) {
        case let (a?, b?):
            let ctx = try! EagerContext()
            let outs = try! Ops(ctx).build("AddV2")
                .addInput(a).addInput(b)
                .attr("T", dtype: Scalar.tfDataType)
                .execute(outputs: 1)
            return .init(Tensor<Scalar>.fromOwnedHandle(outs[0]))
        case let (a?, nil): return .init(a)
        case let (nil, b?): return .init(b)
        default:            return .zero
        }
    }

    @inlinable
    public static func - (lhs: Self, rhs: Self) -> Self {
        switch (lhs.base, rhs.base) {
        case let (a?, b?):
            let ctx = try! EagerContext()
            let outs = try! Ops(ctx).build("Sub")
                .addInput(a).addInput(b)
                .attr("T", dtype: Scalar.tfDataType)
                .execute(outputs: 1)
            return .init(Tensor<Scalar>.fromOwnedHandle(outs[0]))
        case let (a?, nil): return .init(a) // a - 0 == a
        case let (nil, b?): // 0 - b == -b
            let ctx = try! EagerContext()
            let outs = try! Ops(ctx).build("Neg")
                .addInput(b)
                .attr("T", dtype: Scalar.tfDataType)
                .execute(outputs: 1)
            return .init(Tensor<Scalar>.fromOwnedHandle(outs[0]))
        default: return .zero
        }
    }

    // MARK: Equatable  (cheap handle identity is fine for tangents)
    @inlinable
    public static func == (lhs: Self, rhs: Self) -> Bool {
        switch (lhs.base, rhs.base) {
        case (nil, nil):    return true
        case let (a?, b?):  return a.handle == b.handle
        default:            return false
        }
    }

    // MARK: Differentiable
    public typealias TangentVector = _TensorTangent<Scalar>

    @inlinable
    public mutating func move(by direction: TangentVector) {
        self = self + direction
    }
}

/// Make floating tensors differentiable.
/// NOTE: `Tensor` is a class; the requirement is `public func move(by:)` (non-`mutating`).
extension Tensor: Differentiable where Scalar: TensorFlowFloatingPoint {
    public typealias TangentVector = _TensorTangent<Scalar>

    @inlinable
    public func move(by direction: TangentVector) {
        // In-place: self += direction
        guard let d = direction.base else { return }
        let ctx = try! EagerContext()
        let outs = try! Ops(ctx).build("AddV2")
            .addInput(self)
            .addInput(d)
            .attr("T", dtype: Scalar.tfDataType)
            .execute(outputs: 1)

        // Swap underlying handle.
        let newHandle = outs[0]
        TFE_DeleteTensorHandle(self.handle)
        self.handle = newHandle
    }
}
#endif
