import Foundation
import TF4SwiftCore

/// Attempts to read the op list from the TensorFlow C runtime (TF_GetAllOpList).
/// On hosts where this isn’t implemented, we throw and the Pipeline falls back
/// to the bundled ops.pbtxt (see Pipeline.run()).
enum RuntimeOpListLoader {
  static func load() throws -> TFMinimal.OpList {
    // Fetch the serialized tensorflow.OpList from the linked TF runtime
    let data = try TFOpRegistry.runtimeOpListData()
    // Decode minimal OpList (names + inputs + attrs + summary)
    let minimal = try TFMinimal.decodeOpList(from: data)
    return minimal
  }
}
