import Foundation
import CTensorFlow

public enum TFCoreIntrospection {
  public enum Error: Swift.Error { case runtimeUnavailable(String) }

  /// Returns the serialized tensorflow.OpList (protobuf bytes) from the runtime.
  public static func getAllOpListData() throws -> Data {
    guard let buf = TF_GetAllOpList() else {
      throw Error.runtimeUnavailable("TF_GetAllOpList returned nil")
    }
    defer { TF_DeleteBuffer(buf) }

    let length = Int(buf.pointee.length)
    if length == 0 { return Data() }
    guard let base = buf.pointee.data else {
      throw Error.runtimeUnavailable("TF_GetAllOpList buffer has null data pointer")
    }
    return Data(bytes: base, count: length)
  }
}
