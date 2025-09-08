import Foundation
import TF4SwiftCore

/// Internal dynamic runner used by generated wrappers.
/// Delegates to TFExec (Core) which owns the actual eager execution.
@usableFromInline
enum OpRunner {

  @usableFromInline
  static func unary<T>(_ graphOp: String, in ctx: EagerContext, _ x: Tensor<T>) -> Tensor<T> {
    TFExec.unary(graphOp, in: ctx, x)
  }

  @usableFromInline
  static func binary<T>(_ graphOp: String, in ctx: EagerContext, _ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T> {
    TFExec.binary(graphOp, in: ctx, x, y)
  }

  // Attr-forwarding overloads
  @usableFromInline
  static func unary<T>(
    _ graphOp: String,
    in ctx: EagerContext,
    _ x: Tensor<T>,
    intAttrs: [String: Int64] = [:],
    boolAttrs: [String: Bool] = [:],
    stringAttrs: [String: String] = [:],
    intListAttrs: [String: [Int]] = [:]
  ) -> Tensor<T> {
    TFExec.unary(
      graphOp, in: ctx, x,
      intAttrs: intAttrs, boolAttrs: boolAttrs, stringAttrs: stringAttrs, intListAttrs: intListAttrs
    )
  }

  @usableFromInline
  static func binary<T>(
    _ graphOp: String,
    in ctx: EagerContext,
    _ x: Tensor<T>,
    _ y: Tensor<T>,
    intAttrs: [String: Int64] = [:],
    boolAttrs: [String: Bool] = [:],
    stringAttrs: [String: String] = [:],
    intListAttrs: [String: [Int]] = [:]
  ) -> Tensor<T> {
    TFExec.binary(
      graphOp, in: ctx, x, y,
      intAttrs: intAttrs, boolAttrs: boolAttrs, stringAttrs: stringAttrs, intListAttrs: intListAttrs
    )
  }

  @usableFromInline
  static func unary2<T>(_ graphOp: String, in ctx: EagerContext, _ x: Tensor<T>) -> (Tensor<T>, Tensor<T>) {
    TFExec.unary2(graphOp, in: ctx, x)
  }

  @usableFromInline
  static func binary2<T>(_ graphOp: String, in ctx: EagerContext, _ x: Tensor<T>, _ y: Tensor<T>) -> (Tensor<T>, Tensor<T>) {
    TFExec.binary2(graphOp, in: ctx, x, y)
  }
}
