import Testing
import _Differentiation

@testable import TF4SwiftCore
import TF4SwiftOps

@Suite("More AD: sigmoid/tanh")
struct EagerMoreADTests {
    @Test func vjp_sigmoid_center() throws {
        let x = try Tensor<Float>(0.0)

        // Forward + pullback
        let (y, pb) = valueWithPullback(at: x, of: sigmoid)

        // Forward check: sigmoid(0) = 0.5
        #expect(abs(y.scalar - 0.5) < 1e-6)

        // Pullback check: d/dx sigmoid(x) at 0 is 0.25
        let seed = try Tensor<Float>(1.0)
        let g = pb(Tensor<Float>.TangentVector(seed)).base!
        #expect(abs(g.scalar - 0.25) < 1e-6)
    }

    @Test func vjp_tanh_center() throws {
        let x = try Tensor<Float>(0.0)

        // Forward + pullback
        let (y, pb) = valueWithPullback(at: x, of: tanh)

        // Forward check: tanh(0) = 0
        #expect(abs(y.scalar - 0.0) < 1e-6)

        // Pullback check: d/dx tanh(x) at 0 is 1
        let seed = try Tensor<Float>(1.0)
        let g = pb(Tensor<Float>.TangentVector(seed)).base!
        #expect(abs(g.scalar - 1.0) < 1e-6)
    }
}
