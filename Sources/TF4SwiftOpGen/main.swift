// Sources/TF4SwiftOpGen/main.swift
import Foundation

 
@inline(__always)
 func eprint(_ message: String) {
     if let data = (message + "\n").data(using: .utf8) {
         FileHandle.standardError.write(data)
     }
 }


private func run() throws {
    // 1) Resolve ops.pbtxt
    guard let url = findOpsPbtxtInBuildProducts() ?? findOpsPbtxtInSources() else {
        throw NSError(domain: "TF4SwiftOpGen", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "ops.pbtxt not found (looked in build bundle and Sources/TF4SwiftOpGen)."])
    }
    let text = try String(contentsOf: url, encoding: .utf8)
    print("🔎 Using ops.pbtxt at: \(url.path)")

    // 2) Which ops to emit for PR-3
    let wanted = [
        "AddV2", "Mul", "MatMul",
        "Relu", "ReluGrad",
        "Sigmoid", "SigmoidGrad",
        "Tanh", "TanhGrad"
    ]

    // 3) Emit per-domain files under Sources/TF4SwiftOps/Generated/<Domain>/
    let fm = FileManager.default
    var emittedOnce: Set<String> = []

    for name in wanted {
        guard let op = findOp(named: name, in: text) else {
            throw NSError(domain: "TF4SwiftOpGen", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "op not found: \(name)"])
        }
        let (domain, fileName, contents, also) = try emitSwiftForOp(op, in: text)

        // Skip if we've already written this file (e.g. ReluGrad emitted with Relu)
        if emittedOnce.contains(fileName) { continue }
        emittedOnce.insert(fileName)
        also.forEach { emittedOnce.insert($0) }

        let genDir = URL(fileURLWithPath: "Sources/TF4SwiftOps/Generated/\(domain)", isDirectory: true)
        if !fm.fileExists(atPath: genDir.path) {
            try fm.createDirectory(at: genDir, withIntermediateDirectories: true)
        }
        let outURL = genDir.appendingPathComponent(fileName)
        try contents.write(to: outURL, atomically: true, encoding: String.Encoding.utf8)
        print("✅ Generated \(outURL.path)")
    }
}

// Run
do {
    try run()
} catch {
    eprint("❌ tf4swift-opgen error: \(error)\n")
    exit(1)
}

import TF4SwiftCore
// In tf4swift-opgen target:
do {
  let list = try RuntimeOpListLoader.load()
  print("Runtime op count:", list.op.count)
  // e.g., check one or two expected ops:
  let names = Set(list.op.map(\.name))
  print("Has AddV2:", names.contains("AddV2"))
} catch {
  print("Runtime loader failed:", error)
}
