import TF4SwiftCore

/// Root namespace for organized, Swifty ops.
/// Usage:
///   let ops = Ops(context)
///   ops.math  // Math group
///   ops.nn    // NN group
///   ...
@frozen
public struct Ops {
  public let ctx: EagerContext
  @inlinable public init(_ ctx: EagerContext) { self.ctx = ctx }

  @inlinable public var math: Math     { Math(ctx) }
  @inlinable public var nn: NN         { NN(ctx) }
  @inlinable public var array: Array   { Array(ctx) }
  @inlinable public var linalg: Linalg { Linalg(ctx) }
  @inlinable public var image: Image   { Image(ctx) }
  @inlinable public var random: Random { Random(ctx) }
  @inlinable public var io: IO         { IO(ctx) }
  @inlinable public var control: Control { Control(ctx) }
  @inlinable public var other: Other   { Other(ctx) }
}

public extension Ops {
  @frozen struct Math     { @usableFromInline let ctx: EagerContext; @inlinable public init(_ ctx: EagerContext) { self.ctx = ctx } }
  @frozen struct NN       { @usableFromInline let ctx: EagerContext; @inlinable public init(_ ctx: EagerContext) { self.ctx = ctx } }
  @frozen struct Array    { @usableFromInline let ctx: EagerContext; @inlinable public init(_ ctx: EagerContext) { self.ctx = ctx } }
  @frozen struct Linalg   { @usableFromInline let ctx: EagerContext; @inlinable public init(_ ctx: EagerContext) { self.ctx = ctx } }
  @frozen struct Image    { @usableFromInline let ctx: EagerContext; @inlinable public init(_ ctx: EagerContext) { self.ctx = ctx } }
  @frozen struct Random   { @usableFromInline let ctx: EagerContext; @inlinable public init(_ ctx: EagerContext) { self.ctx = ctx } }
  @frozen struct IO       { @usableFromInline let ctx: EagerContext; @inlinable public init(_ ctx: EagerContext) { self.ctx = ctx } }
  @frozen struct Control  { @usableFromInline let ctx: EagerContext; @inlinable public init(_ ctx: EagerContext) { self.ctx = ctx } }
  @frozen struct Other    { @usableFromInline let ctx: EagerContext; @inlinable public init(_ ctx: EagerContext) { self.ctx = ctx } }
}
