// Sources/TF4SwiftCore/Core/Ops+Fill.swift
import CTensorFlow

public extension Ops {
    func zerosLike<T: TensorFlowScalar>(_ x: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
        let outs = try build("ZerosLike")
            .device(device)
            .addInput(x)
            .attr("T", dtype: T.tfDataType)
            .execute(outputs: 1)
        return Tensor<T>.fromOwnedHandle(outs[0])
    }

    func onesLike<T: TensorFlowNumeric>(_ x: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
        let outs = try build("OnesLike")
            .device(device)
            .addInput(x)
            .attr("T", dtype: T.tfDataType)
            .execute(outputs: 1)
        return Tensor<T>.fromOwnedHandle(outs[0])
    }

    func fill<T: TensorFlowScalar>(_ shape: Tensor<Int32>, value: Tensor<T>, device: String? = nil) throws -> Tensor<T> {
        let outs = try build("Fill")
            .device(device)
            .addInput(shape)
            .addInput(value)
            .attr("T", dtype: T.tfDataType)
            .execute(outputs: 1)
        return Tensor<T>.fromOwnedHandle(outs[0])
    }
}

public extension Tensor where Scalar: TensorFlowScalar {
    static func filled(_ value: Scalar, shape: [Int64], in ctx: EagerContext, device: String? = nil) throws -> Tensor<Scalar> {
        let dimsI32: [Int32] = shape.map { Int32($0) }
        let shapeTensor = try Tensor<Int32>.fromArray(dimsI32, shape: [Int64(dimsI32.count)])
        let valueTensor  = try Tensor<Scalar>(value)
        return try Ops(ctx).fill(shapeTensor, value: valueTensor, device: device)
    }
}

public extension Tensor where Scalar: TensorFlowNumeric {
    static func zeros(shape: [Int64], in ctx: EagerContext, device: String? = nil) throws -> Tensor<Scalar> {
        let dimsI32: [Int32] = shape.map { Int32($0) }
        let shapeTensor = try Tensor<Int32>.fromArray(dimsI32, shape: [Int64(dimsI32.count)])
        let zeroScalar: Scalar = .zero
        let valueTensor = try Tensor<Scalar>(zeroScalar)
        return try Ops(ctx).fill(shapeTensor, value: valueTensor, device: device)
    }
}

// Provide two overloads so the '1' literal is always well-typed.
public extension Tensor where Scalar: TensorFlowNumeric & BinaryInteger {
    static func ones(shape: [Int64], in ctx: EagerContext, device: String? = nil) throws -> Tensor<Scalar> {
        let dimsI32: [Int32] = shape.map { Int32($0) }
        let shapeTensor = try Tensor<Int32>.fromArray(dimsI32, shape: [Int64(dimsI32.count)])
        let oneScalar: Scalar = 1
        let valueTensor = try Tensor<Scalar>(oneScalar)
        return try Ops(ctx).fill(shapeTensor, value: valueTensor, device: device)
    }
}

public extension Tensor where Scalar: TensorFlowNumeric & BinaryFloatingPoint {
    static func ones(shape: [Int64], in ctx: EagerContext, device: String? = nil) throws -> Tensor<Scalar> {
        let dimsI32: [Int32] = shape.map { Int32($0) }
        let shapeTensor = try Tensor<Int32>.fromArray(dimsI32, shape: [Int64(dimsI32.count)])
        let oneScalar: Scalar = 1.0
        let valueTensor = try Tensor<Scalar>(oneScalar)
        return try Ops(ctx).fill(shapeTensor, value: valueTensor, device: device)
    }
}
