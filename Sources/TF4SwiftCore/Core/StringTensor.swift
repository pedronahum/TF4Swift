// Sources/TF4SwiftCore/Core/StringTensor.swift
@frozen
public struct StringTensor {
    public let base: Tensor<String>
    public init(_ base: Tensor<String>) { self.base = base }

    // convenience inits
    public init(_ value: String) throws { self.init(try Tensor<String>(value)) }
    public init(_ scalars: [String]) throws { self.init(try Tensor<String>(scalars)) }

    public var scalar: String { base.scalar }
    // public var array: [String] { ... } // add when you add vector readback
}
