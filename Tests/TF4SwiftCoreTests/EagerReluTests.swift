#if canImport(Testing)
import Testing
@testable import TF4SwiftCore
import TF4SwiftOps

@Suite("Relu smoke")
struct EagerReluTests {
    @Test func reluFloats() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let x = try Tensor<Float>([-1, 0, 2])
        let y = try ops.relu(x)
        #expect(y.array == [0, 0, 2])
    }
}
#else
import XCTest
@testable import TF4SwiftCore
import TF4SwiftOps

final class EagerReluTests: XCTestCase {
    func testReluFloats() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let x = try Tensor<Float>([-1, 0, 2])
        let y = try ops.relu(x)
        XCTAssertEqual(y.array, [0, 0, 2])
    }
}
#endif
