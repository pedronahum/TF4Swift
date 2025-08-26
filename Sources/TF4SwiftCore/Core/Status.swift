import CTensorFlow

public final class TFStatus {
    public let ptr: OpaquePointer = TF_NewStatus()!
    public func throwIfError() throws {
        let c = TF_GetCode(ptr)
        if c != TF4SWIFT_OK_CODE() { throw TensorFlowError.status(code: c, message: String(cString: TF_Message(ptr))) }
    }
    deinit { TF_DeleteStatus(ptr) }
}
