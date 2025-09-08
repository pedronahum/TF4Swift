import Foundation

public enum GeneratorError: Error, CustomStringConvertible {
  case runtime(String)
  case io(String)
  case parse(String)

  public var description: String {
    switch self {
    case .runtime(let m): return "Runtime error: \(m)"
    case .io(let m):      return "I/O error: \(m)"
    case .parse(let m):   return "Parse error: \(m)"
    }
  }
}
