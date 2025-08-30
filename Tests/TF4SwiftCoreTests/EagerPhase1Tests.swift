import Testing
@testable import TF4SwiftCore
import TF4SwiftOps

@Suite("Phase 1: creation, devices, basics")
struct EagerPhase1Tests {
    @Test func addVectors1D() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let a = try Tensor<Float>([1,2,3,4])
        let b = try Tensor<Float>([10,20,30,40])
        let c = try ops.add(a, b)
        #expect(c.array == [11,22,33,44])
    }

    @Test func relu1D() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let x = try Tensor<Float>([-1, 0, 2])
        let y = try ops.relu(x)
        #expect(y.array == [0,0,2])
    }

    @Test func onesZerosLike() throws {
        let ctx = try EagerContext()
        let ops = Ops(ctx)
        let x = try Tensor<Float>([0, 0, 0])
        let z = try ops.zerosLike(x)
        let o = try ops.onesLike(x)
        #expect(z.array == [0,0,0])
        #expect(o.array == [1,1,1])
    }

    @Test func deviceListing() throws {
        let ctx = try EagerContext()
        let names = try ctx.deviceNames()
        // CPU should always be present on macOS.
        #expect(names.contains(where: { $0.contains("CPU") }))
    }

    // Tests/TF4SwiftCoreTests/EagerPhase1Tests.swift
    // ...
    @Test func twoDInitializer() {
      let created = (try? Tensor<Float>([[1,2],[3,4]])) != nil
      #expect(created)
    }

}
