import CTensorFlow

/// An existential tensor (type-erased). Useful for heterogenous outputs & tooling.
public protocol AnyTensor {
    var _handle: OpaquePointer { get }         // TFE_TensorHandle*
    var dataType: TensorDataType { get }       // scalar dtype
    // You can add: shape metadata, device, etc. later as needed.
}

extension Tensor: AnyTensor {
    public var _handle: OpaquePointer { handle }
    public var dataType: TensorDataType { .init(Scalar.tfDataType) }
}

/// Factory that turns an owned eager handle + dtype into an `AnyTensor`.
/// You keep ownership transfer semantics confined to this single point.
@usableFromInline
internal func makeTensor(dataType: TensorDataType, owning handle: OpaquePointer) -> AnyTensor {
    switch dataType._cDataType {
    case TF_BOOL:    return Tensor<Bool>.fromOwnedHandle(handle)
    case TF_INT8:    return Tensor<Int8>.fromOwnedHandle(handle)
    case TF_INT16:   return Tensor<Int16>.fromOwnedHandle(handle)
    case TF_INT32:   return Tensor<Int32>.fromOwnedHandle(handle)
    case TF_INT64:   return Tensor<Int64>.fromOwnedHandle(handle)
    case TF_UINT8:   return Tensor<UInt8>.fromOwnedHandle(handle)
    case TF_UINT16:  return Tensor<UInt16>.fromOwnedHandle(handle)
    case TF_UINT32:  return Tensor<UInt32>.fromOwnedHandle(handle)
    case TF_UINT64:  return Tensor<UInt64>.fromOwnedHandle(handle)
    case TF_FLOAT:   return Tensor<Float>.fromOwnedHandle(handle)
    case TF_DOUBLE:  return Tensor<Double>.fromOwnedHandle(handle)
    // case TF_BFLOAT16: return Tensor<BFloat16>.fromOwnedHandle(handle) // when added
    // in AnyTensor.swift (factory)
    case TF_STRING:  return Tensor<String>.fromOwnedHandle(handle)

    default:
        fatalError("Unhandled dtype: \(dataType)")
    }
}
