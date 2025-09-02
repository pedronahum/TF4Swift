import Foundation

/// Group buckets for codegen/output directory names.
public enum ApiDefGroup: String, Sendable {
  case math, nn, array, linalg, image, random, io, control, other

  /// Map an endpoint string like "math.logical_and" → .math
  public static func fromEndpointName(_ endpoint: String) -> ApiDefGroup {
    // Use the prefix before the first dot, lowercased.
    let prefix = endpoint.split(separator: ".", maxSplits: 1).first?.lowercased() ?? ""
    switch prefix {
    case "math":   return .math
    case "nn":     return .nn
    case "array":  return .array
    case "linalg": return .linalg
    case "image":  return .image
    case "random": return .random
    case "io":     return .io
    case "control":return .control
    default:       return .other
    }
  }
}

/// One `op { ... }` record from an `api_def_*.pbtxt` file.
public struct ApiDefRecord: Sendable {
  public var graphOpName: String
  public var endpoints: [String]   // ordered as they appear; first is "canonical"
  public var summary: String?      // optional, single-line summaries only (for now)
}

/// An index for quick lookups by graph op name.
public struct ApiDefIndex: Sendable {
  public let recordsByGraphOpName: [String: ApiDefRecord]

  public init(records: [ApiDefRecord]) {
    var map: [String: ApiDefRecord] = [:]
    for rec in records {
      // Merge endpoints if the same op shows up multiple times (rare).
      if var existing = map[rec.graphOpName] {
        var merged = existing
        var seen = Set(existing.endpoints)
        for e in rec.endpoints where !seen.contains(e) {
          merged.endpoints.append(e)
          seen.insert(e)
        }
        if merged.summary == nil { merged.summary = rec.summary }
        map[rec.graphOpName] = merged
      } else {
        map[rec.graphOpName] = rec
      }
    }
    self.recordsByGraphOpName = map
  }

  /// The endpoints for a given graph op (may be empty if unknown).
  public func endpoints(for graphOpName: String) -> [String] {
    recordsByGraphOpName[graphOpName]?.endpoints ?? []
  }

  /// Pick a group for this graph op.
  /// Rule: if there is any endpoint with a prefix (has a dot), use the first one’s prefix;
  /// else if there’s any endpoint at all, use its prefix (often none) → .other; else → .other.
  public func group(for graphOpName: String) -> ApiDefGroup {
    guard let rec = recordsByGraphOpName[graphOpName] else { return .other }
    if let firstWithPrefix = rec.endpoints.first(where: { $0.contains(".") }) {
      return ApiDefGroup.fromEndpointName(firstWithPrefix)
    }
    if let first = rec.endpoints.first {
      return ApiDefGroup.fromEndpointName(first)
    }
    return .other
  }

  /// The "canonical" public name to expose for this op.
  /// Rule: prefer the first endpoint; otherwise fall back to the graph op name (lowerCamelCase).
  public func canonicalName(for graphOpName: String) -> String {
    if let rec = recordsByGraphOpName[graphOpName], let first = rec.endpoints.first {
      // take the last segment of "pkg.symbol"
      if let last = first.split(separator: ".").last {
        return Self.toLowerCamel(String(last))
      }
      return Self.toLowerCamel(first)
    }
    return Self.toLowerCamel(graphOpName)
  }

  private static func toLowerCamel(_ s: String) -> String {
    guard !s.isEmpty else { return s }
    var out = s
    let first = out.removeFirst()
    return String(first).lowercased() + out
  }
}
