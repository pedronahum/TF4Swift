import TF4SwiftCore

/// Namespace for non-throwing, AD-friendly wrappers.
public enum _Raw {}

/// Minimal runtime helpers local to TF4SwiftOps.
enum _Runtime {
    /// A process-global eager context for _Raw conveniences and AD entry points.
    /// In production you’ll likely want explicit context injection or thread-local contexts.
    static let defaultContext: EagerContext = {
        // If context creation fails, that’s fatal for AD entry points.
        return try! EagerContext()
    }()
}
