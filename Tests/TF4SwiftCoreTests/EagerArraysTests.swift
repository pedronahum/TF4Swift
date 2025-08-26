#if canImport(Testing)
import Testing
@testable import TF4SwiftCore

@Suite("Eager API — vectors & devices")
struct EagerArraysTests {
    @Test("Add two Float vectors")
    func addVectors() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        // Int literals work because of the convenience init; explicit Float is also fine
        let a = try Tensor<Float>([1, 2, 3, 4])
        let b = try Tensor<Float>([10, 20, 30, 40])
        let c = try ops.add(a, b)
        #expect(c.array == [11, 22, 33, 44])
    }

    @Test("List devices always includes CPU")
    func listDevices() throws {
        let ctx = try EagerContext()
        let names = try ctx.deviceNames()
        #expect(names.contains(where: { $0.contains("CPU") }))
    }
}
#else
import XCTest
@testable import TF4SwiftCore

final class EagerArraysTests: XCTestCase {
    func testAddVectors() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let a = try Tensor<Float>([1, 2, 3, 4])
        let b = try Tensor<Float>([10, 20, 30, 40])
        let c = try ops.add(a, b)
        XCTAssertEqual(c.array, [11, 22, 33, 44])
    }

    func testListDevices() throws {
        let ctx = try EagerContext()
        let names = try ctx.deviceNames()
        XCTAssertTrue(names.contains(where: { $0.contains("CPU") }))
    }
}
#endif
