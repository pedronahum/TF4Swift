import Foundation

// Keep CLI error type local.
private enum CLIError: Error {
  case usage(String?)
}

// Execution mode for the tool.
private enum Mode { case help, run(Config) }

// MARK: - CLI parsing

/// Parse CLI args. If the user asked for help, return `.help` (no error).
private func parseArgsOrHelp() throws -> Mode {
  var opsPbtxt: String? = nil
  var apiDefDir: String? = nil
  var outDir: String = "Sources/TF4SwiftOps/Generated"
  var preferRuntime = true
  var verbose = false
  var emitWrappers = false
  var wireRaw = false
  var wireDynamic = false

  var it = CommandLine.arguments.dropFirst().makeIterator()
  while let arg = it.next() {
    switch arg {
    case "--ops-pbtxt":
      guard let p = it.next() else { throw CLIError.usage("--ops-pbtxt requires a path") }
      opsPbtxt = p
      preferRuntime = false
    case "--api-def-dir":
      guard let p = it.next() else { throw CLIError.usage("--api-def-dir requires a path") }
      apiDefDir = p
    case "--out-dir":
      guard let p = it.next() else { throw CLIError.usage("--out-dir requires a path") }
      outDir = p
    case "--no-runtime":
      preferRuntime = false
    case "-v", "--verbose":
      verbose = true
    case "--emit-wrappers":
      emitWrappers = true
    case "--wire-raw":
      wireRaw = true
    case "--wire-dynamic":
      wireDynamic = true
    case "-h", "--help":
      return .help
    default:
      throw CLIError.usage("Unknown argument: \(arg)")
    }
  }

  let cfg = Config(
    opsPbtxtPath: opsPbtxt,
    apiDefDir: apiDefDir,
    outDir: outDir,
    preferRuntime: preferRuntime,
    verbose: verbose,
    emitWrappers: emitWrappers,
    wireRaw: wireRaw,
    wireDynamic: wireDynamic
  )
  return .run(cfg)
}

private func printHelp() {
  let help = """
  Usage:
    swift run -c release tf4swift-opgen [options]

  Options:
    --ops-pbtxt <path>     Use ops.pbtxt at <path> instead of runtime.
    --api-def-dir <path>   Directory containing python_api api_def_*.pbtxt (optional).
    --out-dir <path>       Output directory (default: Sources/TF4SwiftOps/Generated).
    --no-runtime           Do not attempt runtime; use bundled ops.pbtxt (or --ops-pbtxt).
    -v, --verbose          Verbose logs.
    --emit-wrappers        Also generate simple unary/binary wrappers (compilable; not wired yet).
    --wire-raw             Make wrappers call `_Raw.<op>` instead of `__unimplemented(...)`.
    --wire-dynamic         Make wrappers call `OpRunner` (dynamic eager by name) instead of `__unimplemented(...)`.

    -h, --help             Show this help.

  Examples:
    # Runtime + python API endpoints → write plan JSON
    swift run -c release tf4swift-opgen -- \\
      --api-def-dir /path/to/tensorflow/core/api_def/python_api \\
      --out-dir Sources/TF4SwiftOps/Generated

    # Use a specific ops.pbtxt snapshot
    swift run -c release tf4swift-opgen -- \\
      --ops-pbtxt Sources/TF4SwiftOpGen/ops.pbtxt \\
      --out-dir Sources/TF4SwiftOps/Generated
  """
  fputs(help + "\n", stderr)
}

// MARK: - Top-level entry point (no @main)

fputs("[opgen] start\n", stderr)

do {
  switch try parseArgsOrHelp() {
  case .help:
    printHelp()
  case .run(let cfg):
    try Pipeline.run(cfg)
  }
} catch CLIError.usage(let message) {
  if let m = message, !m.isEmpty { fputs(m + "\n", stderr) }
  printHelp()
  exit(2)
} catch {
  fputs("tf4swift-opgen fatal error: \(error)\n", stderr)
  exit(1)
}
