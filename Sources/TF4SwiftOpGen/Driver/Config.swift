import Foundation

public struct Config: Sendable {
  /// If provided, use ops.pbtxt at this path instead of the runtime.
  public var opsPbtxtPath: String?
  /// If provided, scan python api_def files here to enrich endpoints & groups.
  public var apiDefDir: String?
  /// Where to write outputs (default: Sources/TF4SwiftOps/Generated).
  public var outDir: String
  /// Try runtime first? If false and no --ops-pbtxt, we fall back to the bundled ops.pbtxt.
  public var preferRuntime: Bool
  /// Verbose logging to stdout.
  public var verbose: Bool
  /// Emit wrapper functions.
  public var emitWrappers: Bool

  public var wireRaw: Bool

  public var wireDynamic: Bool

  public init(
    opsPbtxtPath: String? = nil,
    apiDefDir: String? = nil,
    outDir: String = "Sources/TF4SwiftOps/Generated",
    preferRuntime: Bool = true,
    verbose: Bool = false,
    emitWrappers: Bool = false,
    wireRaw: Bool = false,
    wireDynamic: Bool = false
  ) {
    self.opsPbtxtPath = opsPbtxtPath
    self.apiDefDir = apiDefDir
    self.outDir = outDir
    self.preferRuntime = preferRuntime
    self.verbose = verbose
    self.emitWrappers = emitWrappers
    self.wireRaw = wireRaw
    self.wireDynamic = wireDynamic
  }
}
