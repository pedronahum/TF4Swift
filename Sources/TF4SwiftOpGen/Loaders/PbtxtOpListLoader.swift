import Foundation

/// Parses a TensorFlow `ops.pbtxt` (textproto) to extract op names (fallback path).
/// We only need top-level `op { name: "..." }` for Commit 2.
public enum PbtxtOpListLoader {

  public static func load(fromPath path: String) throws -> TFMinimal.OpList {
    guard FileManager.default.fileExists(atPath: path) else {
      throw GeneratorError.io("ops.pbtxt not found at \(path)")
    }
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return try load(fromText: text)
  }

  public static func load(fromText text: String) throws -> TFMinimal.OpList {
    var ops: [TFMinimal.OpDef] = []
    let lines = text.split(whereSeparator: \.isNewline).map(String.init)

    var i = 0
    while i < lines.count {
      let s = lines[i].trimmingCharacters(in: .whitespaces)
      if s == "op{" || s.hasPrefix("op {") {
        i += 1
        var depth = 1
        var opName: String?

        while i < lines.count && depth > 0 {
          let raw = lines[i]
          let trimmed = raw.trimmingCharacters(in: .whitespaces)

          if opName == nil, depth == 1, let n = parseNameLine(trimmed) {
            opName = n
          }

          depth += countChar(raw, "{")
          depth -= countChar(raw, "}")
          i += 1
        }

        if let n = opName, !n.isEmpty {
          ops.append(TFMinimal.OpDef(name: n))
        }
        continue
      }
      i += 1
    }

    return TFMinimal.OpList(op: ops)
  }

  private static func countChar(_ s: String, _ ch: Character) -> Int {
    var c = 0; for chx in s where chx == ch { c += 1 }; return c
  }

  /// Match `name: "Something"` only (quotes required).
  private static func parseNameLine(_ line: String) -> String? {
    guard line.hasPrefix("name:") else { return nil }
    guard let q1 = line.firstIndex(of: "\"") else { return nil }
    let rest = line[line.index(after: q1)...]
    guard let q2 = rest.firstIndex(of: "\"") else { return nil }
    return String(rest[..<q2])
  }
}
