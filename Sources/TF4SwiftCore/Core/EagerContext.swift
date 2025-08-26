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


public extension EagerContext {
    func deviceNames() throws -> [String] {
        let st = TFStatus()
        let list = TFE_ContextListDevices(self.ptr, st.ptr)!
        try st.throwIfError()
        defer { TF_DeleteDeviceList(list) }
        let count = TF_DeviceListCount(list)
        var out: [String] = []
        out.reserveCapacity(Int(count))
        for i in 0..<count {
            let cstr = TF_DeviceListName(list, i, st.ptr)
            try st.throwIfError()
            out.append(String(cString: cstr!))
        }
        return out
    }
}
