import Foundation

public struct OpPlanRecord: Codable, Sendable {
  public var graphOpName: String
  public var canonicalName: String
  public var group: String
  public var endpoints: [String]
  public var arity: Int
  public var attrs: [String]
  public var summary: String?
  public var numOutputs: Int
}

public enum Pipeline {
  public static func run(_ cfg: Config) throws {
    // 1) Load OpList (runtime → pbtxt, or forced pbtxt)
    let opList: TFMinimal.OpList

    if let pb = cfg.opsPbtxtPath {
      if cfg.verbose { print("[opgen] Loading OpList from pbtxt: \(pb)") }
      opList = try PbtxtOpListLoader.load(fromPath: pb)
    } else if cfg.preferRuntime {
      do {
        if cfg.verbose { print("[opgen] Loading OpList from runtime (TF_GetAllOpList)") }
        opList = try RuntimeOpListLoader.load()
      } catch {
        if cfg.verbose {
          print("[opgen] Runtime load failed: \(error)\n[opgen] Falling back to bundled ops.pbtxt")
        }
        opList = try loadBundledPbtxt()
      }
    } else {
      if cfg.verbose { print("[opgen] Using bundled ops.pbtxt (preferRuntime=false)") }
      opList = try loadBundledPbtxt()
    }

    let opDefByName: [String: TFMinimal.OpDef] =
      Dictionary(uniqueKeysWithValues: opList.op.map { ($0.name, $0) })
    let opNames = opList.op.map(\.name)
    if cfg.verbose { print("[opgen] Loaded \(opNames.count) ops") }

    // 2) Optionally load python api_def endpoints
    let apiIndex: ApiDefIndex?
    if let dir = cfg.apiDefDir {
      if cfg.verbose { print("[opgen] Scanning python_api api_def at: \(dir)") }
      let recs = try PythonApiDefScanner.load(fromDir: dir)
      if cfg.verbose { print("[opgen] Found \(recs.count) api_def records") }
      apiIndex = ApiDefIndex(records: recs)
    } else {
      apiIndex = nil
    }

    // 3) Build plan → one record per op
    var plan: [OpPlanRecord] = []
    plan.reserveCapacity(opNames.count)

    for name in opNames {
      let endpoints = apiIndex?.endpoints(for: name) ?? []
      let group: String = apiIndex?.group(for: name).rawValue ?? "other"
      let canonical: String = apiIndex?.canonicalName(for: name) ?? toLowerCamel(name)

      // Prefer python_api summary; else runtime OpDef summary (if present)
      let meta = opDefByName[name]
      let arity: Int = meta?.inputArgs.count ?? 0
      let attrNames: [String] = meta?.attrs.map { $0.name } ?? []

      let summary = (apiIndex?.recordsByGraphOpName[name]?.summary?.isEmpty == false)
        ? apiIndex?.recordsByGraphOpName[name]?.summary
        : ((meta?.summary.isEmpty == false) ? meta?.summary : nil)

      plan.append(
        OpPlanRecord(
          graphOpName: name,
          canonicalName: canonical,
          group: group,
          endpoints: endpoints,
          arity: arity,
          attrs: attrNames,
          summary: summary,
          numOutputs: 0 // current generator only needs multi-output for a few known ops handled by name
        )
      )
    }

    // 4) Ensure outDir exists and write plan json
    try ensureDir(cfg.outDir)
    let planPath = (cfg.outDir as NSString).appendingPathComponent("op_plan.json")
    try writeJSON(plan, to: planPath)
    if cfg.verbose { print("[opgen] Wrote plan to \(planPath)") }

    // 5) Emit grouped "Names" indexes (kept unchanged)
    try SwiftEmitter.emit(plan: plan,
                          swiftOutDir: "Sources/TF4SwiftOps/Generated",
                          verbose: cfg.verbose)

    // 6) Emit wrappers
    if cfg.emitWrappers {
      try WrapperEmitter.emit(
        plan: plan,
        groups: WrapperEmitter.defaultGroupsToEmit,
        swiftOutDir: "Sources/TF4SwiftOps/Generated",
        verbose: cfg.verbose,
        wireRaw: cfg.wireRaw,
        wireDynamic: cfg.wireDynamic
      )

      // Emit coverage report based on generated wrappers
      let coveragePath = (cfg.outDir as NSString).appendingPathComponent("OP_COVERAGE.md")
      try CoverageEmitter.emit(
        plan: plan,
        swiftOutDir: "Sources/TF4SwiftOps/Generated",
        to: coveragePath,
        verbose: cfg.verbose
      )
    }

    // 7) Done / stats
    if cfg.verbose {
      let counts = Dictionary(grouping: plan, by: \.group).mapValues { $0.count }
      print("[opgen] Group counts: \(counts.sorted { $0.key < $1.key })")
    }
  }

  // MARK: - Helpers

  private static func ensureDir(_ path: String) throws {
    var isDir: ObjCBool = false
    if !FileManager.default.fileExists(atPath: path, isDirectory: &isDir) {
      try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
    } else if !isDir.boolValue {
      throw GeneratorError.io("Out path exists and is not a directory: \(path)")
    }
  }

  private static func writeJSON<T: Encodable>(_ value: T, to path: String) throws {
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try enc.encode(value)
    try data.write(to: URL(fileURLWithPath: path), options: .atomic)
  }

  private static func toLowerCamel(_ s: String) -> String {
    guard !s.isEmpty else { return s }
    var out = s
    let first = out.removeFirst()
    return String(first).lowercased() + out
  }

  private static func loadBundledPbtxt() throws -> TFMinimal.OpList {
    #if SWIFT_PACKAGE
    guard let url = Bundle.module.url(forResource: "ops", withExtension: "pbtxt") else {
      throw GeneratorError.io("Bundled ops.pbtxt not found in resources")
    }
    let text = try String(contentsOf: url)
    return try PbtxtOpListLoader.load(fromText: text)
    #else
    throw GeneratorError.io("Bundled resources not available (not in SwiftPM context)")
    #endif
  }
}
