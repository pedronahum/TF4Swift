import Testing
@testable import TF4SwiftCore
import TF4SwiftOps

@Suite("Elementwise Mul & MatMul smoke")
struct EagerMulMatMulTests {
    @Test func mulScalars() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let a = try Tensor<Float>(3)
        let b = try Tensor<Float>(4)
        let c = try ops.mul(a, b)
        #expect(c.scalars == [12])
    }

    @Test func mulVectors() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let a = try Tensor<Float>([1,2,3,4])
        let b = try Tensor<Float>([10,20,30,40])
        let c = try ops.mul(a, b)
        #expect(c.scalars == [10,40,90,160])
    }

    @Test func matmul2x3_3x2() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)

        // A: 2x3, B: 3x2
        let A = try Tensor<Float>.fromArray([1,2,3, 4,5,6].map(Float.init), shape: [2,3])
        let B = try Tensor<Float>.fromArray([7,8, 9,10, 11,12].map(Float.init), shape: [3,2])

        let C = try ops.matmul(A, B)
        // Expected C = [[58,64],[139,154]]
        #expect(C.scalars == [58, 64, 139, 154].map(Float.init))
        #expect(try C.shape() == [2,2])
    }

    @Test func matmulTranspose() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)

        // A: 3x2, B: 3x2 -> with transposeB, B^T is 2x3; A (3x2) × B^T (2x3) is invalid.
        // Instead, test transposeA: A^T (2x3) × B (3x2) -> 2x2
        let A = try Tensor<Float>.fromArray([1,2, 3,4, 5,6].map(Float.init), shape: [3,2]) // 3x2
        let B = try Tensor<Float>.fromArray([7,8, 9,10, 11,12].map(Float.init), shape: [3,2]) // 3x2

        let C = try ops.matmul(A, B, transposeA: true) // A^T is 2x3; (2x3)*(3x2) = 2x2
        #expect(try C.shape() == [2,2])
    }
}
