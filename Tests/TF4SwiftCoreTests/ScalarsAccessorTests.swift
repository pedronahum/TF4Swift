import Testing
@testable import TF4SwiftCore

@Suite("Tensor.scalars (row-major, any rank)")
struct ScalarsAccessorTests {
    @Test func oneDScalars() throws {
        let v: [Float] = [1, 2, 3]
        let t = try Tensor<Float>(nested: v)
        #expect(t.scalars == v)
    }

    @Test func twoDScalars() throws {
        let m: [[Float]] = [[1, 2], [3, 4]]
        let t = try Tensor<Float>(nested: m)
        #expect(t.scalars == [1, 2, 3, 4])
    }

    @Test func threeDScalars() throws {
        let c: [[[Float]]] = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ]
        let t = try Tensor<Float>(nested: c)
        #expect(t.scalars == [1, 2, 3, 4, 5, 6, 7, 8])
    }
}
