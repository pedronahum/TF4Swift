#if DEBUG && DEBUG_BROADCAST_CHECKS
import Testing
@testable import TF4SwiftCore

@Suite("Preflight broadcasting (DEBUG_BROADCAST_CHECKS)")
struct BroadcastingPreflightTests {
    @Test func addMismatchedThrows() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)

        // a: [2,3], b: [4]  -> incompatible (3 vs 4)
        let a = try Tensor<Float>.fromArray((0..<6).map { Float($0) }, shape: [2, 3])
        let b = try Tensor<Float>.fromArray([1, 1, 1, 1],                shape: [4])

        #expect(throws: BroadcastError.self) {
            _ = try ops._addWithPreflight(a, b)   // <- use internal helper
        }
    }
}
#endif
