import CTensorFlow

/// Human-readable TensorFlow dtype names for diagnostics.
@inlinable
public func dtypeName(_ dt: TF_DataType) -> String {
    switch dt {
    case TF_BOOL:      return "bool"
    case TF_INT8:      return "int8"
    case TF_INT16:     return "int16"
    case TF_INT32:     return "int32"
    case TF_INT64:     return "int64"
    case TF_UINT8:     return "uint8"
    case TF_UINT16:    return "uint16"
    case TF_UINT32:    return "uint32"
    case TF_UINT64:    return "uint64"
    case TF_BFLOAT16:  return "bfloat16"
    case TF_FLOAT:     return "float32"
    case TF_DOUBLE:    return "float64"
    case TF_STRING:    return "string"
    default:           return "dtype(\(dt.rawValue))"
    }
}

/// Renders `[Int64]` shapes as `[d0,d1,...]`.
@inlinable
public func shapeDescription(_ shape: [Int64]) -> String {
    "[\(shape.map(String.init).joined(separator: ","))]"
}
