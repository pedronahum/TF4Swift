public enum TFExec {
  // Unary
  public static func unary<T>(_ graphOp: String, in ctx: EagerContext, _ x: Tensor<T>) -> Tensor<T> { let b = OpBuilder(ctx: ctx, name: graphOp); b.addInput(x); return b.runOne() }

  public static func unary<T>(
    _ graphOp: String, in ctx: EagerContext, _ x: Tensor<T>,
    intAttrs: [String: Int64] = [:],
    boolAttrs: [String: Bool] = [:],
    stringAttrs: [String: String] = [:],
    intListAttrs: [String: [Int]] = [:]
  ) -> Tensor<T> {
    let b = OpBuilder(ctx: ctx, name: graphOp)
    b.addInput(x)
    for (k, v) in intAttrs     { b.setAttr(k, int: v) }
    for (k, v) in boolAttrs    { b.setAttr(k, bool: v) }
    for (k, v) in stringAttrs  { b.setAttr(k, string: v) }
    for (k, v) in intListAttrs { b.setAttr(k, ints: v.map(Int64.init)) }
    return b.runOne()
  }

  // Binary
  public static func binary<T>(_ graphOp: String, in ctx: EagerContext, _ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T> { let b = OpBuilder(ctx: ctx, name: graphOp); b.addInput(x); b.addInput(y); return b.runOne() }

  public static func binary<T>(
    _ graphOp: String, in ctx: EagerContext, _ x: Tensor<T>, _ y: Tensor<T>,
    intAttrs: [String: Int64] = [:],
    boolAttrs: [String: Bool] = [:],
    stringAttrs: [String: String] = [:],
    intListAttrs: [String: [Int]] = [:]
  ) -> Tensor<T> {
    let b = OpBuilder(ctx: ctx, name: graphOp)
    b.addInput(x); b.addInput(y)
    for (k, v) in intAttrs     { b.setAttr(k, int: v) }
    for (k, v) in boolAttrs    { b.setAttr(k, bool: v) }
    for (k, v) in stringAttrs  { b.setAttr(k, string: v) }
    for (k, v) in intListAttrs { b.setAttr(k, ints: v.map(Int64.init)) }
    return b.runOne()
  }

  // 2-output (same-T)
  public static func unary2<T>(_ graphOp: String, in ctx: EagerContext, _ x: Tensor<T>) -> (Tensor<T>, Tensor<T>) {
    let b = OpBuilder(ctx: ctx, name: graphOp); b.addInput(x); return b.runTwo()
  }
  public static func binary2<T>(_ graphOp: String, in ctx: EagerContext, _ x: Tensor<T>, _ y: Tensor<T>) -> (Tensor<T>, Tensor<T>) {
    let b = OpBuilder(ctx: ctx, name: graphOp); b.addInput(x); b.addInput(y); return b.runTwo()
  }
}
