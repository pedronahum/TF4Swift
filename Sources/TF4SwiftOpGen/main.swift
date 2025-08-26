import Foundation

// ===== Minimal models =====
struct ArgDef { var name = ""; var typeAttr: String?; var typeToken: String?; var numberAttr: String?; var typeListAttr: String? }
struct AttrDef { var name = ""; var type = ""; var allowedTypes: [String] = [] }
struct OpDef  { var name = ""; var inputs: [ArgDef] = []; var outputs: [ArgDef] = []; var attrs: [AttrDef] = [] }

enum ParseError: Error, CustomStringConvertible {
    case fileMissing(String), opNotFound(String), malformed(String)
    var description: String {
        switch self {
        case .fileMissing(let p): return "pbtxt file not found at: \(p)"
        case .opNotFound(let n):  return "op not found: \(n)"
        case .malformed(let m):   return "pbtxt parsing error: \(m)"
        }
    }
}

// ===== ops.pbtxt locator =====
func locateOpsPbtxt() throws -> URL {
    let fm = FileManager.default
    let env = ProcessInfo.processInfo.environment
    if let p = env["TF_OPS_PBTXT"] {
        let u = URL(fileURLWithPath: p)
        if fm.fileExists(atPath: u.path) { return u }
        throw ParseError.fileMissing("TF_OPS_PBTXT set but file not found: \(p)")
    }
    #if SWIFT_PACKAGE
    if let u = Bundle.module.url(forResource: "ops", withExtension: "pbtxt") { return u }
    #endif
    let cwd = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
    let underSources = cwd.appendingPathComponent("Sources/TF4SwiftOpGen/ops.pbtxt")
    if fm.fileExists(atPath: underSources.path) { return underSources }
    let here = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("ops.pbtxt")
    if fm.fileExists(atPath: here.path) { return here }
    throw ParseError.fileMissing("Set TF_OPS_PBTXT, or place ops.pbtxt under Sources/TF4SwiftOpGen/")
}

func readText(_ url: URL) throws -> String {
    let s = try String(contentsOf: url, encoding: .utf8)
    return s.replacingOccurrences(of: "\r\n", with: "\n")
}

// ===== Quote-aware brace matching =====
func findMatchingBrace(in text: String, from open: String.Index) -> String.Index? {
    precondition(text[open] == "{")
    var i = open, depth = 0, inString = false, escaped = false
    while i < text.endIndex {
        let ch = text[i]
        if inString {
            if escaped { escaped = false }
            else if ch == "\\" { escaped = true }
            else if ch == "\"" { inString = false }
        } else {
            if ch == "\"" { inString = true }
            else if ch == "{" { depth += 1 }
            else if ch == "}" {
                depth -= 1
                if depth == 0 { return i }
            }
        }
        i = text.index(after: i)
    }
    return nil
}

// ===== Scan all top-level op blocks =====
func scanOpBlocks(_ s: String) -> [String] {
    var blocks: [String] = []
    var i = s.startIndex
    var inString = false
    var escaped = false

    func skipWS(_ j: inout String.Index) {
        while j < s.endIndex, s[j].isWhitespace { j = s.index(after: j) }
    }

    while i < s.endIndex {
        let ch = s[i]
        if inString {
            if escaped { escaped = false }
            else if ch == "\\" { escaped = true }
            else if ch == "\"" { inString = false }
            i = s.index(after: i)
            continue
        }
        if ch == "\"" { inString = true; i = s.index(after: i); continue }

        // look for 'op' token
        if ch == "o" {
            let n1 = s.index(after: i)
            if n1 < s.endIndex, s[n1] == "p" {
                let beforeOK = (i == s.startIndex) || !s[s.index(before: i)].isLetter
                var j = s.index(after: n1)
                skipWS(&j)
                if j < s.endIndex, s[j] == ":" { j = s.index(after: j); skipWS(&j) }
                if beforeOK, j < s.endIndex, s[j] == "{" {
                    let open = j
                    if let close = findMatchingBrace(in: s, from: open) {
                        let inner = String(s[s.index(after: open)..<close])
                        blocks.append(inner)
                        i = s.index(after: close)
                        continue
                    } else { break }
                }
            }
        }
        i = s.index(after: i)
    }
    return blocks
}

// ===== Helpers to parse sub-blocks =====
func firstQuoted(_ key: String, in s: String) -> String? {
    let ns = s as NSString
    let pattern = "(?m)\\b" + NSRegularExpression.escapedPattern(for: key) + "\\s*:\\s*\"([^\"]+)\""
    let re = try! NSRegularExpression(pattern: pattern)
    if let m = re.firstMatch(in: s, range: NSRange(location: 0, length: ns.length)), m.numberOfRanges >= 2 {
        return ns.substring(with: m.range(at: 1))
    }
    return nil
}
func firstUnquotedToken(_ key: String, in s: String) -> String? {
    let ns = s as NSString
    let pattern = "(?m)\\b" + NSRegularExpression.escapedPattern(for: key) + "\\s*:\\s*([A-Za-z0-9_]+)"
    let re = try! NSRegularExpression(pattern: pattern)
    if let m = re.firstMatch(in: s, range: NSRange(location: 0, length: ns.length)), m.numberOfRanges >= 2 {
        return ns.substring(with: m.range(at: 1))
    }
    return nil
}
func allBlocks(named key: String, in s: String) -> [String] {
    var out: [String] = []
    var i = s.startIndex
    while i < s.endIndex {
        guard let r = s[i...].range(of: key) else { break }
        var j = r.upperBound
        while j < s.endIndex, s[j].isWhitespace { j = s.index(after: j) }
        if j < s.endIndex, s[j] == ":" { j = s.index(after: j) }
        while j < s.endIndex, s[j].isWhitespace { j = s.index(after: j) }
        guard j < s.endIndex, s[j] == "{" else { i = r.upperBound; continue }
        guard let close = findMatchingBrace(in: s, from: j) else { break }
        out.append(String(s[s.index(after: j)..<close]))
        i = s.index(after: close)
    }
    return out
}
func parseArg(_ b: String) -> ArgDef {
    var a = ArgDef()
    a.name         = firstQuoted("name", in: b) ?? a.name
    a.typeAttr     = firstQuoted("type_attr", in: b)
    a.numberAttr   = firstQuoted("number_attr", in: b)
    a.typeListAttr = firstQuoted("type_list_attr", in: b)
    a.typeToken    = firstUnquotedToken("type", in: b)
    return a
}
func parseAttr(_ b: String) -> AttrDef {
    var a = AttrDef()
    a.name = firstQuoted("name", in: b) ?? a.name
    a.type = firstQuoted("type", in: b) ?? a.type
    let ns = b as NSString
    let allowPattern = "allowed_values\\s*\\{\\s*list\\s*\\{([\\s\\S]*?)\\}\\s*\\}"
    if let re = try? NSRegularExpression(pattern: allowPattern),
       let m = re.firstMatch(in: b, range: NSRange(location: 0, length: ns.length)) {
        let listText = ns.substring(with: m.range(at: 1))
        let typeRe = try! NSRegularExpression(pattern: "(?:^|\\s)type\\s*:\\s*([A-Za-z0-9_]+)")
        let lns = listText as NSString
        for m2 in typeRe.matches(in: listText, range: NSRange(location: 0, length: lns.length)) {
            a.allowedTypes.append(lns.substring(with: m2.range(at: 1)))
        }
    }
    return a
}
func parseOpBlock(_ s: String) -> OpDef {
    var op = OpDef()
    op.name    = firstQuoted("name", in: s) ?? ""
    op.inputs  = allBlocks(named: "input_arg",  in: s).map(parseArg)
    op.outputs = allBlocks(named: "output_arg", in: s).map(parseArg)
    op.attrs   = allBlocks(named: "attr",       in: s).map(parseAttr)
    return op
}
func findOp(named target: String, in full: String) -> OpDef? {
    for block in scanOpBlocks(full) {
        if let nm = firstQuoted("name", in: block), nm == target { return parseOpBlock(block) }
    }
    return nil
}

// ===== Emitters =====
func emitBinarySameType(_ op: OpDef, methodName: String, opName: String, usesNumeric: Bool) throws -> String {
    guard op.inputs.count == 2, op.outputs.count == 1 else {
        throw ParseError.malformed("\(opName) expected 2 inputs, 1 output")
    }
    guard let taX = op.inputs[0].typeAttr, let taY = op.inputs[1].typeAttr, taX == taY,
          let taOut = op.outputs[0].typeAttr, taOut == taX else {
        throw ParseError.malformed("\(opName) inputs/outputs must share the same type_attr")
    }
    let bound = usesNumeric ? "TensorFlowNumeric" : "TensorFlowScalar"
    return """
    // ---------------------------------------------------------------------------
    //  AUTO-GENERATED FILE. DO NOT EDIT.
    //  Generated by TF4SwiftOpGen from ops.pbtxt (op: \(opName))
    // ---------------------------------------------------------------------------

    import TF4SwiftCore

    public extension Ops {
        /// Generated wrapper for TensorFlow \(opName).
        func \(methodName)<T: \(bound)>(_ x: Tensor<T>, _ y: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
            let outs = try build("\(opName)")
                .device(device)
                .addInput(x)
                .addInput(y)
                .attr("T", dtype: T.tfDataType)
                .execute(outputs: 1)
            return Tensor<T>.fromOwnedHandle(outs[0])
        }
    }
    """
}
func emitUnarySameType(_ op: OpDef, methodName: String, opName: String, usesNumeric: Bool) throws -> String {
    guard op.inputs.count == 1, op.outputs.count == 1 else {
        throw ParseError.malformed("\(opName) expected 1 input, 1 output")
    }
    guard let taIn = op.inputs[0].typeAttr, let taOut = op.outputs[0].typeAttr, taIn == taOut else {
        throw ParseError.malformed("\(opName) input/output must share the same type_attr")
    }
    let bound = usesNumeric ? "TensorFlowNumeric" : "TensorFlowScalar"
    return """
    // ---------------------------------------------------------------------------
    //  AUTO-GENERATED FILE. DO NOT EDIT.
    //  Generated by TF4SwiftOpGen from ops.pbtxt (op: \(opName))
    // ---------------------------------------------------------------------------

    import TF4SwiftCore

    public extension Ops {
        /// Generated wrapper for TensorFlow \(opName).
        func \(methodName)<T: \(bound)>(_ x: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
            let outs = try build("\(opName)")
                .device(device)
                .addInput(x)
                .attr("T", dtype: T.tfDataType)
                .execute(outputs: 1)
            return Tensor<T>.fromOwnedHandle(outs[0])
        }
    }
    """
}
func emitUnaryRaw(_ opName: String, method: String, floatOnly: Bool) -> String {
    let bound = floatOnly ? "TensorFlowFloatingPoint" : "TensorFlowScalar"
    return """
    import TF4SwiftCore

    public extension _Raw {
        /// Non-throwing wrapper for TensorFlow \(opName).
        static func \(method)<T: \(bound)>(features x: Tensor<T>, device: String? = nil) -> Tensor<T> {
            let ctx = _Runtime.defaultContext
            let outs = try! Ops(ctx).build("\(opName)")
                .device(device)
                .addInput(x)
                .attr("T", dtype: T.tfDataType)
                .execute(outputs: 1)
            return Tensor<T>.fromOwnedHandle(outs[0])
        }
    }
    """
}
func emitBinaryRaw(_ opName: String, method: String, floatOnly: Bool) -> String {
    let bound = floatOnly ? "TensorFlowFloatingPoint" : "TensorFlowScalar"
    return """
    import TF4SwiftCore

    public extension _Raw {
        /// Non-throwing wrapper for TensorFlow \(opName).
        static func \(method)<T: \(bound)>(_ x: Tensor<T>, _ y: Tensor<T>, device: String? = nil) -> Tensor<T> {
            let ctx = _Runtime.defaultContext
            let outs = try! Ops(ctx).build("\(opName)")
                .device(device)
                .addInput(x)
                .addInput(y)
                .attr("T", dtype: T.tfDataType)
                .execute(outputs: 1)
            return Tensor<T>.fromOwnedHandle(outs[0])
        }
    }
    """
}
// AD glue is gated; enable with -DTF4SWIFT_EXPERIMENTAL_AD once Tensor conforms to Differentiable.
func emitReluAD() -> String {
    return """
    #if canImport(_Differentiation)
    import _Differentiation
    import TF4SwiftCore

    @inlinable
    @differentiable(reverse, wrt: x)
    public func relu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
        _Raw.relu(features: x)
    }

    @inlinable
    @derivative(of: relu)
    public func _vjpRelu<T: TensorFlowFloatingPoint>(
        _ x: Tensor<T>
    ) -> (
        value: Tensor<T>,
        pullback: (Tensor<T>.TangentVector) -> Tensor<T>.TangentVector
    ) {
        let y = _Raw.relu(features: x)
        return (y, { v in
            guard let seed = v.base else { return .zero }
            let g = _Raw.reluGrad(gradients: seed, features: x)
            return Tensor<T>.TangentVector(g)
        })
    }
    #endif
    """
}


// ===== Utilities =====
func findPackageRoot(startingAt start: URL) -> URL? {
    let fm = FileManager.default
    var dir = start
    while true {
        if fm.fileExists(atPath: dir.appendingPathComponent("Package.swift").path) { return dir }
        let parent = dir.deletingLastPathComponent()
        if parent.path == dir.path { return nil }
        dir = parent
    }
}

// ===== Main =====
do {
    let pbtxtURL = try locateOpsPbtxt()
    print("🔎 Using ops.pbtxt at: \(pbtxtURL.path)")
    let text = try readText(pbtxtURL)

    // ---- AddV2 (or Add fallback) ----
    let addOpName = (findOp(named: "AddV2", in: text) != nil) ? "AddV2" : "Add"
    guard let addOp = findOp(named: addOpName, in: text) else { throw ParseError.opNotFound("AddV2 / Add") }
    let addSrc = try emitBinarySameType(addOp, methodName: "add", opName: addOpName, usesNumeric: true)

    // ---- Relu ----
    guard let reluOp = findOp(named: "Relu", in: text) else { throw ParseError.opNotFound("Relu") }
    // Ops.relu (throwing), float-only bound
    let reluOpsExt = try emitUnarySameType(reluOp, methodName: "relu", opName: "Relu", usesNumeric: false)
        .replacingOccurrences(of: "TensorFlowScalar", with: "TensorFlowFloatingPoint")
    // _Raw.relu (non-throwing)
    let reluRaw = emitUnaryRaw("Relu", method: "relu", floatOnly: true)

    // ---- ReluGrad (for VJP) ----
    guard let reluGradOp = findOp(named: "ReluGrad", in: text) else {
        throw ParseError.opNotFound("ReluGrad")
    }
    let reluGradOps = try emitBinarySameType(
        reluGradOp,                      // <- use reluGradOp here
        methodName: "reluGrad",
        opName: "ReluGrad",
        usesNumeric: false
    ).replacingOccurrences(of: "TensorFlowScalar", with: "TensorFlowFloatingPoint")

    // _Raw.reluGrad (non-throwing)
    let reluGradRaw =
    """
    import TF4SwiftCore

    public extension _Raw {
        /// Non-throwing wrapper for TensorFlow ReluGrad.
        static func reluGrad<T: TensorFlowFloatingPoint>(gradients: Tensor<T>, features: Tensor<T>, device: String? = nil) -> Tensor<T> {
            let ctx = _Runtime.defaultContext
            let outs = try! Ops(ctx).build("ReluGrad")
                .device(device)
                .addInput(gradients)
                .addInput(features)
                .attr("T", dtype: T.tfDataType)
                .execute(outputs: 1)
            return Tensor<T>.fromOwnedHandle(outs[0])
        }
    }
    """

    // Gated AD glue
    let reluAD = emitReluAD()

    // ---- Write files ----
    let fm = FileManager.default
    let cwd = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
    let root = findPackageRoot(startingAt: cwd) ?? cwd
    let genDir = root.appendingPathComponent("Sources/TF4SwiftOps/Generated", isDirectory: true)
    try fm.createDirectory(at: genDir, withIntermediateDirectories: true)

    let addURL  = genDir.appendingPathComponent("AddV2.swift")
    let reluURL = genDir.appendingPathComponent("Relu.swift")
    try addSrc.write(to: addURL, atomically: true, encoding: .utf8)
    try (reluOpsExt + "\n\n" + reluRaw + "\n\n" + reluGradOps + "\n\n" + reluGradRaw + "\n\n" + reluAD)
        .write(to: reluURL, atomically: true, encoding: .utf8)

    print("✅ Generated \(addURL.path)")
    print("✅ Generated \(reluURL.path)")
} catch {
    fputs("❌ \(error)\n", stderr)
    exit(1)
}
