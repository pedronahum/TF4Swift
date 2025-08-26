import CTensorFlow

/// Public wrapper for TensorFlow C dtypes (no CTensorFlow leak in APIs).
public struct TensorDataType: Equatable, Hashable, CustomStringConvertible {
    public var _cDataType: TF_DataType
    @inlinable public init(_ c: TF_DataType) { self._cDataType = c }

    public var description: String {
        switch _cDataType {
        case TF_BOOL:    return "Bool"
        case TF_INT8:    return "Int8"
        case TF_INT16:   return "Int16"
        case TF_INT32:   return "Int32"
        case TF_INT64:   return "Int64"
        case TF_UINT8:   return "UInt8"
        case TF_UINT16:  return "UInt16"
        case TF_UINT32:  return "UInt32"
        case TF_UINT64:  return "UInt64"
        case TF_FLOAT:   return "Float"
        case TF_DOUBLE:  return "Double"
        case TF_BFLOAT16:return "BFloat16"
        case TF_STRING:  return "String"
        default:         return "TF_DataType(\(_cDataType.rawValue))"
        }
    }
}

/// Types that can report a TensorFlow dtype.
public protocol _TensorFlowDataTypeCompatible {
    static var tensorFlowDataType: TensorDataType { get }
}

/// Scalar types supported by TensorFlow.
public protocol TensorFlowScalar: _TensorFlowDataTypeCompatible {}

/// Numeric scalars (good default for most math ops).
public typealias TensorFlowNumeric = TensorFlowScalar & Numeric

// put near TensorFlowNumeric etc.
public typealias TensorFlowFloatingPoint = TensorFlowScalar & BinaryFloatingPoint


/// Signed numeric scalars.
public typealias TensorFlowSignedNumeric = TensorFlowScalar & SignedNumeric

/// Integer scalars.
public typealias TensorFlowInteger = TensorFlowScalar & BinaryInteger

/// Indices used by TF ops (e.g. Gather/Scatter often allow {Int32, Int64}).
public protocol TensorFlowIndex: TensorFlowInteger {}
extension Int32: TensorFlowIndex {}
extension Int64: TensorFlowIndex {}

// MARK: - Conformances (extend when you add more scalar models)

extension Bool:   TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_BOOL) } }
extension Int8:   TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_INT8) } }
extension Int16:  TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_INT16) } }
extension Int32:  TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_INT32) } }
extension Int64:  TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_INT64) } }
extension UInt8:  TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_UINT8) } }
extension UInt16: TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_UINT16) } }
extension UInt32: TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_UINT32) } }
extension UInt64: TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_UINT64) } }

extension Float:  TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_FLOAT) } }
extension Double: TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_DOUBLE) } }

// If/when you add a BFloat16 scalar model:
// public struct BFloat16 { public var bits: UInt16; public init(_ bits: UInt16) { self.bits = bits } }
// extension BFloat16: TensorFlowScalar { public static var tensorFlowDataType: TensorDataType { .init(TF_BFLOAT16) } }

// NOTE: We intentionally do NOT conform `String` here yet.
public extension TensorFlowScalar {
    @inlinable static var tfDataType: TF_DataType { tensorFlowDataType._cDataType }
}
