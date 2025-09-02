import Foundation
import TF4SwiftCore
import SwiftProtobuf

/// Loads the registered TensorFlow operation list from the linked runtime
/// by calling into TF4SwiftCore (which encapsulates all CTensorFlow usage).
public enum RuntimeOpListLoader {
  /// Returns a decoded minimal OpList (names only for now).
  public static func load() throws -> TFMinimal.OpList {
    let data = try TFOpRegistry.runtimeOpListData()
    var list = TFMinimal.OpList()
    do {
      try list.merge(serializedData: data)
    } catch {
      throw GeneratorError.parse("Failed to decode tensorflow.OpList from runtime: \(error)")
    }
    return list
  }

  /// Optional: dump the raw binary proto for debugging.
  public static func dumpBinary(to path: String) throws {
    let data = try TFOpRegistry.runtimeOpListData()
    try data.write(to: URL(fileURLWithPath: path), options: .atomic)
  }
}
