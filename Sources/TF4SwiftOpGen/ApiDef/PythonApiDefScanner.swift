import Foundation

/// Scanner for TensorFlow's python_api api_def files:
///   tensorflow/core/api_def/python_api/api_def_*.pbtxt
///
/// We read:
///   op {
///     graph_op_name: "LogicalAnd"
///     endpoint { name: "math.logical_and" }
///     endpoint { name: "logical_and" }
///     summary: "Returns the truth value of x AND y element-wise."
///   }
///
/// Multi-line `description: <<END ... END` is intentionally ignored in Week 1.
public enum PythonApiDefScanner {

  public enum ScanError: Error, CustomStringConvertible {
    case notADirectory(String)
    case ioError(String, underlying: Error)

    public var description: String {
      switch self {
      case .notADirectory(let p): return "Not a directory: \(p)"
      case .ioError(let p, let e): return "I/O error reading \(p): \(e)"
      }
    }
  }

  /// Load all `api_def_*.pbtxt` files under `dir` (non-recursive is typical, but we support recursive).
  public static func load(fromDir dir: String, recursive: Bool = true) throws -> [ApiDefRecord] {
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: dir, isDirectory: &isDir), isDir.boolValue else {
      throw ScanError.notADirectory(dir)
    }

    var paths: [String] = []
    if recursive {
      let en = FileManager.default.enumerator(atPath: dir)
      while let entry = en?.nextObject() as? String {
        if entry.hasPrefix("api_def_") && entry.hasSuffix(".pbtxt") {
          paths.append((dir as NSString).appendingPathComponent(entry))
        }
      }
    } else {
      let entries = try FileManager.default.contentsOfDirectory(atPath: dir)
      for entry in entries where entry.hasPrefix("api_def_") && entry.hasSuffix(".pbtxt") {
        paths.append((dir as NSString).appendingPathComponent(entry))
      }
    }

    var out: [ApiDefRecord] = []
    out.reserveCapacity(paths.count)

    for path in paths {
      do {
        let text = try String(contentsOfFile: path, encoding: .utf8)
        if let rec = parseOneFile(text: text) {
          out.append(rec)
        }
      } catch {
        throw ScanError.ioError(path, underlying: error)
      }
    }
    return out
  }

  /// Parse a single api_def_*.pbtxt file (usually contains one `op { ... }`).
  /// Returns nil if no `op` block is present.
  public static func parseOneFile(text: String) -> ApiDefRecord? {
    let lines = text.split(whereSeparator: \.isNewline).map(String.init)
    var i = 0
    let n = lines.count

    var inOp = false
    var inEndpoint = false
    var depth = 0

    var graphOpName: String?
    var endpoints: [String] = []
    var summary: String?

    while i < n {
      let raw = lines[i]
      let line = raw.trimmingCharacters(in: .whitespaces)

      // Enter an op-block
      if !inOp && isBlockStart(line, label: "op") {
        inOp = true
        depth = 1
        i += 1
        continue
      }

      if inOp {
        // Emerging endpoint block?
        if !inEndpoint && isBlockStart(line, label: "endpoint") && depth == 1 {
          inEndpoint = true
        }

        if depth == 1 {
          if graphOpName == nil, let v = parseQuoted(line, key: "graph_op_name") {
            graphOpName = v
          } else if summary == nil, let v = parseQuoted(line, key: "summary") {
            summary = v
          }
        }

        if inEndpoint, let v = parseQuoted(line, key: "name") {
          endpoints.append(v)
        }
      }

      // Update depth after parsing the line
      depth += countChar(raw, "{")
      depth -= countChar(raw, "}")

      // Exit endpoint when block depth falls back to 1
      if inOp, inEndpoint, depth == 1 {
        inEndpoint = false
      }

      // Exit op when block depth falls back to 0
      if inOp, depth == 0 {
        inOp = false
        // Build record (if we got a name)
        if let op = graphOpName, !op.isEmpty {
          return ApiDefRecord(graphOpName: op, endpoints: endpoints, summary: summary)
        } else {
          return nil
        }
      }

      i += 1
    }

    return nil
  }

  // MARK: - Helpers

  private static func isBlockStart(_ line: String, label: String) -> Bool {
    // Accept "label {", "label{", "label: {", "label:{"
    if line == "\(label){" { return true }
    if line.hasPrefix("\(label) {") { return true }
    if line.hasPrefix("\(label):{") { return true }
    if line.hasPrefix("\(label): {") { return true }
    return false
  }

  /// Parse `key: "value"` and return value if present on this line.
  private static func parseQuoted(_ line: String, key: String) -> String? {
    guard line.hasPrefix("\(key):") else { return nil }
    guard let firstQuote = line.firstIndex(of: "\"") else { return nil }
    let rest = line[line.index(after: firstQuote)...]
    guard let secondQuote = rest.firstIndex(of: "\"") else { return nil }
    return String(rest[..<secondQuote])
  }

  private static func countChar(_ s: String, _ ch: Character) -> Int {
    var c = 0; for x in s where x == ch { c += 1 }; return c
  }
}
