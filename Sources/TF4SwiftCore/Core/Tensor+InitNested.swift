import CTensorFlow

// MARK: - Errors

/// Shape/element errors raised during nested-array initialization.
public enum ShapeError: Error, CustomStringConvertible, Sendable {
    /// The nested array is ragged (inner shapes differ).
    case ragged(expected: [Int64], got: [Int64])
    /// A leaf element did not match the tensor Scalar type.
    case elementTypeMismatch(expected: Any.Type, found: Any.Type)

    public var description: String {
        switch self {
        case let .ragged(expected, got):
            return "Ragged nested array. Expected inner shape \(expected), got \(got)."
        case let .elementTypeMismatch(expected, found):
            return "Element type mismatch. Expected elements of type \(expected), found \(found)."
        }
    }
}

// MARK: - Flatten + shape inference (runtime, works for any nesting depth)

@usableFromInline
func _flattenAndInferShape<Scalar>(
    _ value: Any,
    into scalars: inout [Scalar]
) throws -> [Int64] {
    // Leaf element that must be exactly Scalar.
    if let s = value as? Scalar {
        scalars.append(s)
        return [] // leaf contributes no further dims
    }

    // Otherwise we expect a collection (Array).
    let mirror = Mirror(reflecting: value)
    guard mirror.displayStyle == .collection else {
        throw ShapeError.elementTypeMismatch(expected: Scalar.self, found: type(of: value))
    }

    var childShape: [Int64]? = nil
    var count: Int64 = 0
    for child in mirror.children {
        let shape = try _flattenAndInferShape(child.value, into: &scalars)
        if let c = childShape {
            if c != shape {
                throw ShapeError.ragged(expected: c, got: shape)
            }
        } else {
            childShape = shape
        }
        count += 1
    }
    // If the collection is empty, childShape stays nil; treat inner as [].
    return [count] + (childShape ?? [])
}

// MARK: - Public ND initializer

public extension Tensor where Scalar: TensorFlowScalar {
    /// Create a tensor from a nested Swift array of matching `Scalar` elements.
    ///
    /// Examples:
    /// ```
    /// let a = try Tensor<Float>(nested: [1.0, 2.0, 3.0])           // shape [3]
    /// let b = try Tensor<Float>(nested: [[1.0, 2.0], [3.0, 4.0]])   // shape [2, 2]
    /// let c = try Tensor<Float>(nested: [[[1,2],[3,4]], [[5,6],[7,8]]].map { $0.map { $0.map(Float.init) } })
    /// ```
    ///
    /// - Notes:
    ///   - Elements must be **exactly** `Scalar` (no implicit Int→Float coercions).
    ///   - Arrays must be **rectangular** (non‑ragged) at every nesting depth.
    convenience init(nested value: Any) throws {
        var flat: [Scalar] = []
        let shape = try _flattenAndInferShape(value, into: &flat)

        let elementBytes = MemoryLayout<Scalar>.stride
        let totalBytes   = elementBytes * flat.count

        // Allocate TF_Tensor and copy bytes (if any).
        let tfTensor: OpaquePointer = shape.withUnsafeBufferPointer { dims -> OpaquePointer in
            let t = TF_AllocateTensor(Scalar.tfDataType, dims.baseAddress, Int32(dims.count), totalBytes)!
            if totalBytes > 0 {
                _ = flat.withUnsafeBytes { raw in
                    memcpy(TF_TensorData(t), raw.baseAddress!, raw.count)
                }
            }
            return t
        }

        // Wrap in eager handle.
        let st = TFStatus()
        let h  = TFE_NewTensorHandle(tfTensor, st.ptr)!
        TF_DeleteTensor(tfTensor)
        try st.throwIfError()

        self.init(ownedHandle: h)
    }
}
