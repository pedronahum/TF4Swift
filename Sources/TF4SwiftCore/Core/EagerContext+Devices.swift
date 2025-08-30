import CTensorFlow

public extension EagerContext {
    func deviceNames() throws -> [String] {
        let st = TFStatus()
        let listOpt = TFE_ContextListDevices(ptr, st.ptr)
        try st.throwIfError()
        guard let list = listOpt else { return [] }
        defer { TF_DeleteDeviceList(list) }

        let count = TF_DeviceListCount(list)
        var names: [String] = []
        names.reserveCapacity(Int(count))

        for i in 0..<count {
            if let cname = TF_DeviceListName(list, i, st.ptr) {
                try st.throwIfError()
                names.append(String(cString: cname))
            } else {
                try st.throwIfError()
            }
        }
        return names
    }
}
