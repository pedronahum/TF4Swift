import Testing
@testable import TF4SwiftCore

@Suite("ND Array Initializers (shape inference, ragged checks)")
struct NDInitializersTests {
    @Test func oneD() throws {
        // Explicit Float-typed array to avoid Double default in `Any` context
        let v: [Float] = [1, 2, 3]
        let t = try Tensor<Float>(nested: v)
        #expect(t.array == [1, 2, 3])
    }

    @Test func twoD() {
        // Explicit Float at both nesting levels
        let m: [[Float]] = [[1, 2], [3, 4]]
        // Just validate construction succeeds; avoid `array` (rank > 1)
        #expect((try? Tensor<Float>(nested: m)) != nil)
    }

    @Test func threeD() {
        let c: [[[Float]]] = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ]
        // Validate construction succeeds; avoid `array` (rank > 1)
        #expect((try? Tensor<Float>(nested: c)) != nil)
    }

    @Test func raggedThrows() {
        let ragged: [[Float]] = [[1, 2], [3]]
        #expect(throws: ShapeError.self) {
            _ = try Tensor<Float>(nested: ragged)
        }
    }

    @Test func elementTypeMismatchThrows() {
        // Int elements but expecting Float leaves → should fail
        let ints: [[Int]] = [[1, 2], [3, 4]]
        #expect(throws: ShapeError.self) {
            _ = try Tensor<Float>(nested: ints)
        }
    }
}
