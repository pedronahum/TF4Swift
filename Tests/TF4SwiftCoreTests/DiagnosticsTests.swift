import Testing
@testable import TF4SwiftCore
import TF4SwiftOps   // use a generated op so we go through OpBuilder

@Suite("Diagnostics — enriched TF status messages")
struct DiagnosticsTests {
    @Test func addMismatchShowsOpName() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)

        // a: [2,3], b: [4]  -> incompatible (3 vs 4)
        let a = try Tensor<Float>.fromArray((0..<6).map { Float($0) }, shape: [2, 3])
        let b = try Tensor<Float>.fromArray([1, 1, 1, 1],            shape: [4])

        do {
            _ = try ops.add(a, b)   // generated AddV2 wrapper -> OpBuilder.execute
            #expect(Bool(false), "expected AddV2 to fail on incompatible shapes")
        } catch let TensorFlowError.status(_, message) {
            // Message should include op name; device is best effort.
            #expect(message.contains("AddV2"))
            // Optional: check some part of TF's original text or device tag if you like.
        } catch {
            #expect(Bool(false), "unexpected error type: \(error)")
        }
    }
}
