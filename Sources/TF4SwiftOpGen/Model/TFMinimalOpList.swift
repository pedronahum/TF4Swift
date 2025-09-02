import Foundation
import SwiftProtobuf

/// Minimal protobuf types to decode TensorFlow's `tensorflow.OpList` with only the fields we need.
/// We purposely avoid committing large generated files and only implement the essentials.
///
/// message OpList { repeated OpDef op = 1; }
/// message OpDef  { optional string name = 1; /* ignoring the rest for now */ }
public enum TFMinimal {
  public struct OpDef: SwiftProtobuf.Message, Hashable {
    public var name: String = ""
    public var unknownFields = SwiftProtobuf.UnknownStorage()
    public init() {}
    public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
      while let f = try decoder.nextFieldNumber() {
        switch f {
        case 1: try decoder.decodeSingularStringField(value: &name)
        default: try decoder.skipField()
        }
      }
    }
    public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
      if !name.isEmpty { try visitor.visitSingularStringField(value: name, fieldNumber: 1) }
      try unknownFields.traverse(visitor: &visitor)
    }
    public static let protoMessageName = "tensorflow.OpDef"
  }

  public struct OpList: SwiftProtobuf.Message, Hashable {
    public var op: [OpDef] = []
    public var unknownFields = SwiftProtobuf.UnknownStorage()
    public init() {}
    public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
      while let f = try decoder.nextFieldNumber() {
        switch f {
        case 1: try decoder.decodeRepeatedMessageField(value: &op)
        default: try decoder.skipField()
        }
      }
    }
    public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
      if !op.isEmpty { try visitor.visitRepeatedMessageField(value: op, fieldNumber: 1) }
      try unknownFields.traverse(visitor: &visitor)
    }
    public static let protoMessageName = "tensorflow.OpList"
  }
}
