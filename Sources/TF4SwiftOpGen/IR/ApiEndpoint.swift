public enum ApiGrouping {
  public static func group(fromEndpoint fqName: String) -> OpIR.Group {
    if let prefix = fqName.split(separator: ".").first?.lowercased() {
      switch prefix {
      case "math":   return .math
      case "nn":     return .nn
      case "linalg": return .linalg
      case "image":  return .image
      case "array":  return .array
      case "random": return .random
      case "io":     return .io
      case "control":return .control
      default:       break
      }
    }
    return .other
  }
}
