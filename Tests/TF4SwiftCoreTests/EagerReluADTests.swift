import Testing
import _Differentiation
@testable import TF4SwiftCore
import TF4SwiftOps

@Suite("ReLU AD")
struct EagerReluADTests {
    @Test func vjp_relu() throws {
        // x = [-1, 0, 2]
        let x = try Tensor<Float>([-1.0, 0.0, 2.0])

        // Forward + pullback
        let (y, pb) = valueWithPullback(at: x, of: relu)

        // Forward check
        #expect(y.array == [0.0, 0.0, 2.0])

        // Upstream cotangent (seed): ones
        let seed = try Tensor<Float>([1.0, 1.0, 1.0])

        // IMPORTANT: pullback takes a TangentVector, not a Tensor.
        let gradTV = pb(seed.tangentVector)


        // Unwrap the underlying tensor and check its values: [0, 0, 1]
        let grad = gradTV.base!
        #expect(grad.array == [0.0, 0.0, 1.0])
    }
}
