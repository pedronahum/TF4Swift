#if canImport(Testing)
import Testing
@testable import TF4SwiftCore

@Suite("Eager API — scalars")
struct EagerSmokeTests {
    @Test("Add two Float scalars")
    func addFloatScalars() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let a = try Tensor<Float>(3.0)
        let b = try Tensor<Float>(4.0)
        let c = try ops.add(a, b)
        #expect(c.scalar == 7.0)
    }

    @Test("Add two Int32 scalars")
    func addInt32Scalars() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let a = try Tensor<Int32>(21)
        let b = try Tensor<Int32>(21)
        let c = try ops.add(a, b)
        #expect(c.scalar == 42)
    }
}
#else
import XCTest
@testable import TF4SwiftCore

final class EagerSmokeTests: XCTestCase {
    func testAddFloatScalars() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let a = try Tensor<Float>(3.0)
        let b = try Tensor<Float>(4.0)
        let c = try ops.add(a, b)
        XCTAssertEqual(c.scalar, 7.0)
    }

    func testAddInt32Scalars() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let a = try Tensor<Int32>(21)
        let b = try Tensor<Int32>(21)
        let c = try ops.add(a, b)
        XCTAssertEqual(c.scalar, 42)
    }
}
#endif
