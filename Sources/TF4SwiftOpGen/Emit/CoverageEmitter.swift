import Foundation

/// Scans emitted wrapper files and produces a human‑readable coverage report.
/// We deliberately scan generated Swift files instead of trusting plan.arity,
/// because TFMinimal.OpDef may not include inputArgs/outputArgs.
enum CoverageEmitter {

  struct Counts { var unary = 0; var binary = 0 }

  static func emit(
    plan: [OpPlanRecord],
    swiftOutDir: String,
    to path: String,
    verbose: Bool = false
  ) throws {
    // Selection rules should mirror WrapperEmitter
    let selected = plan.filter {
      WrapperEmitter.defaultGroupsToEmit.contains($0.group) ||
      WrapperEmitter.alwaysEmitOps.contains($0.graphOpName)
    }

    let groups = ["Math","NN","Linalg","Array","Image","Random","IO","Control","Other"]

    var totalUnary = 0
    var totalBinary = 0
    var perGroup: [String: Counts] = [:]

    // Scan the generated files for function signatures.
    for g in groups {
      let p = (swiftOutDir as NSString).appendingPathComponent("\(g)/\(g)_Wrappers.swift")
      guard FileManager.default.fileExists(atPath: p) else { continue }
      let text = (try? String(contentsOfFile: p)) ?? ""
      let c = scanFile(text)
      totalUnary += c.unary
      totalBinary += c.binary
      perGroup[g] = c
    }

    // Build report
    let emitted = totalUnary + totalBinary
    let notSelected = plan.count - selected.count
    let skippedSelected = max(0, selected.count - emitted)

    var md: [String] = []
    md.append("# TF4Swift – Wrapper Coverage\n")
    md.append("- **Total TF ops**: \(plan.count)")
    md.append("- **Selected for wrappers**: \(selected.count)")
    md.append("  - Emitted unary: \(totalUnary)")
    md.append("  - Emitted binary: \(totalBinary)")
    md.append("- **Not selected**: \(notSelected)\n")

    md.append("## Emitted by group")
    for g in groups {
      let c = perGroup[g] ?? Counts()
      md.append("- \(g): \(c.unary + c.binary) (unary \(c.unary), binary \(c.binary))")
    }

    // Optional: show a few examples of selected-but-not-emitted (by canonical name)
    let emittedNames = collectFunctionNames(from: swiftOutDir, groups: groups)
    let selectedNames = Set(selected.map { $0.canonicalName })
    let missing = Array(selectedNames.subtracting(emittedNames)).sorted().prefix(16)
    md.append("\n## Skipped (selected but not emitted)")
    md.append("- Count: \(skippedSelected)")
    if !missing.isEmpty {
      md.append("- Examples: " + missing.joined(separator: ", "))
    }

    try md.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    if verbose { print("[opgen] Coverage → \(path)") }
  }

  // Count unary/binary wrappers by looking at function headers.
  private static func scanFile(_ s: String) -> Counts {
    var c = Counts()
    s.enumerateLines { line, _ in
      guard line.contains("func "), line.contains("Tensor<") else { return }
      if line.contains(", _ y: Tensor<") {
        c.binary += 1
      } else if line.contains("(_ x: Tensor<") {
        c.unary += 1
      }
    }
    return c
  }

  private static func collectFunctionNames(from swiftOutDir: String, groups: [String]) -> Set<String> {
    var names = Set<String>()
    let regex = try? NSRegularExpression(pattern: #"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)\s*<"#)
    for g in groups {
      let p = (swiftOutDir as NSString).appendingPathComponent("\(g)/\(g)_Wrappers.swift")
      guard FileManager.default.fileExists(atPath: p),
            let text = try? String(contentsOfFile: p)
      else { continue }
      if let rx = regex {
        let ns = text as NSString
        for m in rx.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
          if m.numberOfRanges > 1 {
            names.insert(ns.substring(with: m.range(at: 1)))
          }
        }
      }
    }
    return names
  }
}
