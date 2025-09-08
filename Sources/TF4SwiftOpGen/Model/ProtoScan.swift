import Foundation

// Minimal protobuf wire parser for TensorFlow OpList/OpDef.
// We only read: OpList.op (field 1: length-delimited OpDef)
// OpDef fields we care about:
//   1: name (string)
//   2: repeated input_arg (length-delimited ArgDef)  -> count them
//   3: repeated output_arg (length-delimited ArgDef) -> count them

struct ProtoScan {
  struct OpShape: Sendable {
    var name: String
    var inputCount: Int
    var outputCount: Int
  }

  static func parseOpList(_ data: Data) throws -> [OpShape] {
    var r = Reader(data)
    var out: [OpShape] = []
    while !r.isAtEnd {
      let (fn, wt) = try r.readKey()
      guard fn == 1, wt == .len else {
        try r.skip(wt) // skip unexpected fields at top-level
        continue
      }
      // Read an OpDef message (length-delimited)
      let sub = try r.readSubmessage()
      var def = OpShape(name: "", inputCount: 0, outputCount: 0)
      var rr = Reader(sub)
      while !rr.isAtEnd {
        let (f, w) = try rr.readKey()
        switch (f, w) {
        case (1, .len): // name
          def.name = try rr.readString()
        case (2, .len): // input_arg (ArgDef)
          _ = try rr.readBytes() // skip ArgDef payload
          def.inputCount += 1
        case (3, .len): // output_arg (ArgDef)
          _ = try rr.readBytes()
          def.outputCount += 1
        default:
          try rr.skip(w)
        }
      }
      out.append(def)
    }
    return out
  }

  // MARK: - Wire reader

  enum Wire: UInt8 {
    case varint = 0
    case i64 = 1
    case len = 2
    case sg = 3 // start group (unused)
    case eg = 4 // end group (unused)
    case i32 = 5
  }

  struct Reader {
    private let bytes: [UInt8]
    private(set) var idx: Int = 0
    init(_ d: Data) { self.bytes = Array(d) }

    var isAtEnd: Bool { idx >= bytes.count }

    mutating func readKey() throws -> (Int, Wire) {
      let key = try readVarint()
      let fieldNumber = Int(key >> 3)
      guard let wire = Wire(rawValue: UInt8(key & 0x7)) else { throw err("bad wire") }
      return (fieldNumber, wire)
    }

    mutating func readVarint() throws -> UInt64 {
      var shift: UInt64 = 0
      var v: UInt64 = 0
      while true {
        guard idx < bytes.count else { throw err("eof varint") }
        let b = bytes[idx]; idx += 1
        v |= UInt64(b & 0x7f) << shift
        if (b & 0x80) == 0 { return v }
        shift += 7
        if shift > 63 { throw err("varint overflow") }
      }
    }

    mutating func readBytes() throws -> Data {
      let len = Int(try readVarint())
      guard idx + len <= bytes.count else { throw err("eof bytes") }
      let d = Data(bytes[idx ..< idx+len])
      idx += len
      return d
    }

    mutating func readString() throws -> String {
      let d = try readBytes()
      return String(decoding: d, as: UTF8.self)
    }

    mutating func readSubmessage() throws -> Data { try readBytes() }

    mutating func skip(_ w: Wire) throws {
      switch w {
      case .varint: _ = try readVarint()
      case .i64:    idx += 8
      case .len:    _ = try readBytes()
      case .i32:    idx += 4
      case .sg, .eg: break // not used in modern protos
      }
      if idx > bytes.count { throw err("skip overflow") }
    }

    func err(_ s: String) -> NSError { NSError(domain: "ProtoScan", code: 1, userInfo: [NSLocalizedDescriptionKey: s]) }
  }
}
