import Foundation

/// Temporary helper used by generated wrappers until we wire to `_Raw`/dynamic execution.
/// Having this keeps the build green while the public API is stabilized.
/// We'll remove these calls in the next commit.
@inline(never)
public func __unimplemented(_ feature: String, file: StaticString = #fileID, line: UInt = #line) -> Never {
  fatalError("[TF4Swift] Not wired yet: \(feature). This wrapper compiles intentionally; it will be connected in the next commit.", file: file, line: line)
}
