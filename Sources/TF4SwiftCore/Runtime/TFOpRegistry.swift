import Foundation
import CTensorFlow

/// Public accessor for TensorFlow's registered operation list from the **linked runtime**.
/// This keeps all CTensorFlow usage encapsulated inside TF4SwiftCore.
///
/// Under the hood, this calls `TF_GetAllOpList()` from the TF C API and returns the
/// serialized `tensorflow.OpList` (binary protobuf).
///
/// See: c_api.h (TF_GetAllOpList)
///  - "Get the OpList of all OpDefs defined in this address space."
///  - Returns a TF_Buffer with serialized OpList; caller must delete the buffer.
///
/// NOTE: We expose raw `Data` here so higher layers (like the OpGen) can decode it
/// with a protobuf implementation of their choice without duplicating C-interop.
public enum TFOpRegistry {
  public struct RuntimeError: Error, CustomStringConvertible {
    public var message: String
    public var description: String { "TensorFlow runtime error: \(message)" }
    public init(_ msg: String) { self.message = msg }
  }

  /// Returns the serialized `tensorflow.OpList` (binary protobuf) for the current process.
  @inlinable
  public static func runtimeOpListData() throws -> Data {
    guard let buf = TF_GetAllOpList() else {
      throw RuntimeError("TF_GetAllOpList() returned a null buffer")
    }
    defer { TF_DeleteBuffer(buf) }

    let length = Int(buf.pointee.length)
    guard length > 0, let base = buf.pointee.data else {
      // Some TF builds may return an empty buffer on error; treat as error for clarity.
      throw RuntimeError("TF_GetAllOpList() returned an empty buffer")
    }
    return Data(bytes: base, count: length)
  }
}
