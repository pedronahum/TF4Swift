import Foundation
import TF4SwiftCore

/// Loads the registered TensorFlow operation list from the linked runtime
/// by calling into TF4SwiftCore (which encapsulates all CTensorFlow usage).
public enum RuntimeOpListLoader {
  /// Returns a decoded minimal OpList (names only).
  public static func load() throws -> TFMinimal.OpList {
    let data = try TFOpRegistry.runtimeOpListData()
    return try TFMinimal.decodeOpList(from: data)
  }

  /// Optional: dump the raw binary proto for debugging.
  public static func dumpBinary(to path: String) throws {
    let data = try TFOpRegistry.runtimeOpListData()
    try data.write(to: URL(fileURLWithPath: path), options: .atomic)
  }
}
