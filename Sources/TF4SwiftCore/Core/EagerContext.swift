import CTensorFlow


// Concurrency: we treat EagerContext as safe to use across threads in this package.
extension EagerContext: @unchecked Sendable {}

public final class EagerContext {
    public let ptr: OpaquePointer
    public init() throws {
        let opts = TFE_NewContextOptions()!
        defer { TFE_DeleteContextOptions(opts) }
        let st = TFStatus()
        ptr = TFE_NewContext(opts, st.ptr)!
        try st.throwIfError()
    }
    deinit { TFE_DeleteContext(ptr) }
}

