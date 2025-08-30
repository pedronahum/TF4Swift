import Testing
@testable import TF4SwiftCore

@Suite("Broadcasting utilities & eager smoke")
struct BroadcastingTests {
    @Test func functionValid() throws {
        let out = try broadcastedShape([3, 1, 4], [1, 5, 4])
        #expect(out == [3, 5, 4])
    }

    @Test func functionInvalid() {
      #expect(throws: BroadcastError.self) {
        _ = try broadcastedShape([2, 3], [4])  // incompatible: 3 vs 4
      }
    }

    #expect(throws: BroadcastError.self) {
      _ = try broadcastedShape([3, 1, 4], [3, 5])  // incompatible: 4 vs 5
    }

    @Test func eagerAddBroadcastShape() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)

        // a: [3, 1, 4]
        let a = try Tensor<Float>.fromArray(
            (0..<12).map { Float($0) }, shape: [3, 1, 4]
        )
        // b: [1, 5, 4]
        let b = try Tensor<Float>.fromArray(
            Array(repeating: 1.0 as Float, count: 20), shape: [1, 5, 4]
        )
        let c = try ops.add(a, b)
        let s = try c.shape()
        #expect(s == [3, 5, 4])
    }
}
