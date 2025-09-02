import Foundation

/// Minimal types + a tiny Protobuf wire decoder to read the TF Op registry emitted
/// by TF_GetAllOpList(). We only decode:
///   OpList  -> repeated OpDef (field 1, length‑delimited)
///   OpDef   -> name (field 1, length‑delimited string)
///
/// This avoids depending on SwiftProtobuf for a single read path and is stable
/// across SwiftProtobuf releases.
public enum TFMinimal {

  // Public data model used by the generator pipeline
  public struct OpDef: Sendable {
    public var name: String
  }

  public struct OpList: Sendable {
    public var op: [OpDef]
  }

  // Parse entry points
  public enum DecodeError: Error, CustomStringConvertible, Sendable {
    case truncated
    case malformedTag
    case invalidWireType(UInt8)
    case lengthOverflow
    case invalidUTF8

    public var description: String {
      switch self {
      case .truncated: return "protobuf data truncated"
      case .malformedTag: return "malformed protobuf tag"
      case .invalidWireType(let w): return "invalid protobuf wire type: \(w)"
      case .lengthOverflow: return "length-delimited field length overflow"
      case .invalidUTF8: return "invalid UTF-8 string in OpDef.name"
      }
    }
  }

  /// Decode a tensorflow.OpList (names only).
  public static func decodeOpList(from data: Data) throws -> OpList {
    var r = WireReader(data: data)
    var ops: [OpDef] = []

    while !r.eof {
      let tag = try r.readVarint()
      guard tag != 0 else { throw DecodeError.malformedTag }
      let fieldNumber = Int(tag >> 3)
      let wireType = UInt8(tag & 0x07)

      if fieldNumber == 1, wireType == 2 {
        let msg = try r.readLengthDelimited()
        let op = try decodeOpDef(from: msg)
        ops.append(op)
      } else {
        try r.skipField(wireType: wireType)
      }
    }

    return OpList(op: ops)
  }

  /// Decode a tensorflow.OpDef, extracting only `name` (field 1).
  public static func decodeOpDef(from data: Data) throws -> OpDef {
    var r = WireReader(data: data)
    var name: String = ""

    while !r.eof {
      let tag = try r.readVarint()
      guard tag != 0 else { throw DecodeError.malformedTag }
      let fieldNumber = Int(tag >> 3)
      let wireType = UInt8(tag & 0x07)

      if fieldNumber == 1, wireType == 2 {
        let bytes = try r.readLengthDelimited()
        guard let s = String(data: bytes, encoding: .utf8) else {
          throw DecodeError.invalidUTF8
        }
        name = s
      } else {
        try r.skipField(wireType: wireType)
      }
    }

    return OpDef(name: name)
  }

  // MARK: - Minimal Protobuf wire reader

  /// Tiny Protobuf wire-format reader (varints, length-delimited, skipping).
  struct WireReader {
    let data: Data
    var index: Int = 0

    var eof: Bool { index >= data.count }

    mutating func readByte() throws -> UInt8 {
      guard index < data.count else { throw DecodeError.truncated }
      defer { index += 1 }
      return data[index]
    }

    mutating func readVarint() throws -> UInt64 {
      var result: UInt64 = 0
      var shift: UInt64 = 0
      while true {
        let b = try readByte()
        result |= UInt64(b & 0x7f) << shift
        if (b & 0x80) == 0 { break }
        shift += 7
        if shift > 63 { throw DecodeError.malformedTag }
      }
      return result
    }

    mutating func readLengthDelimited() throws -> Data {
      let len64 = try readVarint()
      guard len64 <= UInt64(Int.max) else { throw DecodeError.lengthOverflow }
      let len = Int(len64)
      guard index + len <= data.count else { throw DecodeError.truncated }
      let start = index
      index += len
      return data.subdata(in: start..<(start + len))
    }

    mutating func skipField(wireType: UInt8) throws {
      switch wireType {
      case 0: // varint
        _ = try readVarint()
      case 1: // 64-bit
        try skipBytes(8)
      case 2: // length-delimited
        let len64 = try readVarint()
        guard len64 <= UInt64(Int.max) else { throw DecodeError.lengthOverflow }
        try skipBytes(Int(len64))
      case 3: // start group (deprecated) – skip nested until end group
        try skipGroup()
      case 4: // end group – signal to caller; here we treat as no-op
        return
      case 5: // 32-bit
        try skipBytes(4)
      default:
        throw DecodeError.invalidWireType(wireType)
      }
    }

    mutating func skipGroup() throws {
      while true {
        let tag = try readVarint()
        let wt = UInt8(tag & 0x07)
        if wt == 4 { return }           // end group
        try skipField(wireType: wt)
      }
    }

    mutating func skipBytes(_ n: Int) throws {
      guard index + n <= data.count else { throw DecodeError.truncated }
      index += n
    }
  }
}
