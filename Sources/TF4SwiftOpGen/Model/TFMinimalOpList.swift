import Foundation

/// Minimal types + a tiny Protobuf wire decoder to read the TF Op registry emitted
/// by TF_GetAllOpList(). We now decode enough of OpDef to get:
///   - name (field 1)
///   - input_arg (field 2) -> ArgDef { name (1), type (3), type_attr (4), number_attr (5), type_list_attr (6) }
///   - attr (field 4) -> AttrDef { name (1), type (2) }  (we ignore default_value etc.)
///   - summary (field 5)
///
/// Field numbers are taken from TensorFlow's op_def.proto (compatible in TensorBoard compat proto).
/// See: tensorboard/compat/proto/op_def.proto. 
/// (We only rely on the field numbers; we do not need the full schema.)
public enum TFMinimal {

  // MARK: - Public data model

  public struct ArgDef: Sendable {
    public var name: String
    public var type: Int32?         // numeric DataType enum value (if present)
    public var typeAttr: String?     // e.g. "T"
    public var numberAttr: String?   // e.g. "N"
    public var typeListAttr: String? // e.g. "T"
    public init(name: String = "", type: Int32? = nil, typeAttr: String? = nil,
                numberAttr: String? = nil, typeListAttr: String? = nil) {
      self.name = name; self.type = type; self.typeAttr = typeAttr
      self.numberAttr = numberAttr; self.typeListAttr = typeListAttr
    }
  }

  public struct AttrDef: Sendable {
    public var name: String
    public var type: String
    public init(name: String = "", type: String = "") { self.name = name; self.type = type }
  }

  public struct OpDef: Sendable {
    public var name: String
    public var inputArgs: [ArgDef]
    public var attrs: [AttrDef]
    public var summary: String
    public init(name: String,
                inputArgs: [ArgDef] = [],
                attrs: [AttrDef] = [],
                summary: String = "") {
      self.name = name; self.inputArgs = inputArgs; self.attrs = attrs; self.summary = summary
    }
  }

  public struct OpList: Sendable {
    public var op: [OpDef]
    public init(op: [OpDef]) { self.op = op }
  }

  // MARK: - Parse entry points

  public enum DecodeError: Error, CustomStringConvertible, Sendable {
    case truncated, malformedTag, invalidWireType(UInt8), lengthOverflow, invalidUTF8
    public var description: String {
      switch self {
      case .truncated: return "protobuf data truncated"
      case .malformedTag: return "malformed protobuf tag"
      case .invalidWireType(let w): return "invalid protobuf wire type: \(w)"
      case .lengthOverflow: return "length-delimited field length overflow"
      case .invalidUTF8: return "invalid UTF-8 string"
      }
    }
  }

  /// Decode a tensorflow.OpList (names + inputs + attrs + summary).
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

  /// Decode a tensorflow.OpDef (subset we care about).
  public static func decodeOpDef(from data: Data) throws -> OpDef {
    var r = WireReader(data: data)
    var name = ""
    var inputs: [ArgDef] = []
    var attrs: [AttrDef] = []
    var summary = ""

    while !r.eof {
      let tag = try r.readVarint()
      guard tag != 0 else { throw DecodeError.malformedTag }
      let fn = Int(tag >> 3)
      let wt = UInt8(tag & 0x07)
      switch (fn, wt) {
      case (1, 2): // name
        name = try r.readString()
      case (2, 2): // input_arg
        let d = try r.readLengthDelimited()
        inputs.append(try decodeArgDef(from: d))
      case (4, 2): // attr
        let d = try r.readLengthDelimited()
        attrs.append(try decodeAttrDef(from: d))
      case (5, 2): // summary
        summary = try r.readString()
      default:
        try r.skipField(wireType: wt)
      }
    }
    return OpDef(name: name, inputArgs: inputs, attrs: attrs, summary: summary)
  }

  /// Decode ArgDef (subset).
  private static func decodeArgDef(from data: Data) throws -> ArgDef {
    var r = WireReader(data: data)
    var out = ArgDef()
    while !r.eof {
      let tag = try r.readVarint()
      guard tag != 0 else { throw DecodeError.malformedTag }
      let fn = Int(tag >> 3)
      let wt = UInt8(tag & 0x07)
      switch (fn, wt) {
      case (1, 2): out.name = try r.readString()     // name
      case (3, 0): out.type = Int32(bitPattern: UInt32(try r.readVarint())) // DataType enum
      case (4, 2): out.typeAttr = try r.readString()
      case (5, 2): out.numberAttr = try r.readString()
      case (6, 2): out.typeListAttr = try r.readString()
      default:
        try r.skipField(wireType: wt)
      }
    }
    return out
  }

  /// Decode AttrDef (subset).
  private static func decodeAttrDef(from data: Data) throws -> AttrDef {
    var r = WireReader(data: data)
    var name = ""
    var type = ""
    while !r.eof {
      let tag = try r.readVarint()
      guard tag != 0 else { throw DecodeError.malformedTag }
      let fn = Int(tag >> 3)
      let wt = UInt8(tag & 0x07)
      switch (fn, wt) {
      case (1, 2): name = try r.readString() // name
      case (2, 2): type = try r.readString() // type (string, list(type), int, etc.)
      // (3) default_value, (4) description, (5/6/7) constraints — skipped
      default: try r.skipField(wireType: wt)
      }
    }
    return AttrDef(name: name, type: type)
  }

  // MARK: - Minimal Protobuf wire reader

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

    mutating func readString() throws -> String {
      let d = try readLengthDelimited()
      guard let s = String(data: d, encoding: .utf8) else { throw DecodeError.invalidUTF8 }
      return s
    }

    mutating func skipField(wireType: UInt8) throws {
      switch wireType {
      case 0: _ = try readVarint()
      case 1: try skipBytes(8)
      case 2:
        let len64 = try readVarint()
        guard len64 <= UInt64(Int.max) else { throw DecodeError.lengthOverflow }
        try skipBytes(Int(len64))
      case 3: try skipGroup()        // start group (deprecated)
      case 4: return                 // end group
      case 5: try skipBytes(4)
      default: throw DecodeError.invalidWireType(wireType)
      }
    }

    mutating func skipGroup() throws {
      while true {
        let tag = try readVarint()
        let wt = UInt8(tag & 0x07)
        if wt == 4 { return }
        try skipField(wireType: wt)
      }
    }

    mutating func skipBytes(_ n: Int) throws {
      guard index + n <= data.count else { throw DecodeError.truncated }
      index += n
    }
  }
}
