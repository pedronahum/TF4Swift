import Foundation

/// Best‑effort categorization for TensorFlow ops when python_api api_def
/// does not specify a group. Keep this conservative: exact names first,
/// then very obvious prefixes/suffixes.
enum HeuristicGrouper {
  static func group(for name: String) -> String {
    let n = name

    // --- NN / layers / activations ---
    if nnExact.contains(n)
      || n.hasPrefix("Conv") || n.hasPrefix("DepthwiseConv")
      || n.hasSuffix("Pool") || n.contains("BatchNorm")
    { return "nn" }

    // --- Linear algebra / decompositions / FFT ---
    if linalgExact.contains(n)
      || n == "MatMul" || n.hasPrefix("BatchMatMul")
      || n.hasPrefix("Matrix")
      || n == "Cholesky" || n == "Qr" || n == "Svd"
      || n.hasPrefix("FFT") || n.hasPrefix("IFFT")
    { return "linalg" }

    // --- Array / shape / indexing ---
    if arrayExact.contains(n)
      || n.hasPrefix("Concat") || n.hasPrefix("Split")
      || n.hasPrefix("Gather") || n.hasPrefix("Scatter")
      || n == "Reshape" || n == "Transpose"
      || n == "Squeeze" || n == "ExpandDims"
      || n == "Pad" || n == "PadV2" || n.hasSuffix("Slice")
      || n == "Tile" || n.hasPrefix("Reverse")
      || n == "Rank" || n == "Shape" || n == "ShapeN" || n == "Size"
      || n == "Unique" || n == "UniqueV2"
      || n.hasPrefix("TopK") || n == "Where" || n == "OneHot"
    { return "array" }

    // --- Image / resize / color space / codecs ---
    if imageExact.contains(n)
      || n.hasPrefix("Resize")
      || n.hasPrefix("RGBTo") || n.hasPrefix("HSVTo")
      || n.hasPrefix("Decode") || n.hasPrefix("Encode")
      || n.contains("NonMaxSuppression")
    { return "image" }

    // --- Random / sampling ---
    if randomExact.contains(n)
      || n.hasPrefix("Random") || n.hasPrefix("StatelessRandom")
      || n == "TruncatedNormal" || n == "Multinomial"
      || n == "ParameterizedTruncatedNormal"
    { return "random" }

    // --- Control flow ---
    if controlExact.contains(n)
      || ["Switch","Merge","Enter","Exit","NextIteration","LoopCond",
          "If","While","Case","Identity","NoOp","StopGradient","PreventGradient"].contains(n)
    { return "control" }

    // --- IO / datasets / reading / saving ---
    if ioExact.contains(n)
      || n.hasSuffix("Dataset") || n.contains("Reader") || n.contains("Queue")
      || ["ReadFile","WriteFile","Save","Restore","SaveV2","RestoreV2"].contains(n)
    { return "io" }

    // --- Math / elementwise / logical ---
    if mathExact.contains(n)
      || ["Add","AddV2","Sub","Mul","Div","RealDiv","Pow","SquaredDifference",
          "Maximum","Minimum","Mod","FloorMod","Square","Sqrt","Rsqrt","Exp","Expm1",
          "Log","Log1p","Sin","Cos","Tan","Tanh","Asin","Acos","Atan","Sinh","Cosh",
          "Asinh","Acosh","Atanh","Erf","Erfc","Lgamma","Digamma","Abs","Neg","Sign",
          "Round","Rint","Ceil","Floor","IsFinite","IsInf","IsNan","Atan2",
          "Equal","NotEqual","Less","LessEqual","Greater","GreaterEqual",
          "LogicalAnd","LogicalOr","LogicalNot",
          "Real","Imag","Complex","Conj","ComplexAbs"].contains(n)
    { return "math" }

    return "other"
  }

  // --- Exact-name allowlists (non-exhaustive; extend incrementally) ---

  static let nnExact: Set<String> = [
    "Relu","Relu6","LeakyRelu","Elu","Selu","Sigmoid","Softplus","Softsign",
    "Softmax","LogSoftmax","LRN","L2Loss",
    "BiasAdd","Conv2D","Conv3D","DepthwiseConv2dNative",
    "AvgPool","AvgPool3D","MaxPool","MaxPool3D","Dilation2D",
    "FusedBatchNorm","FusedBatchNormV2","FusedBatchNormV3"
  ]

  static let linalgExact: Set<String> = [
    "MatMul","BatchMatMul","BatchMatMulV2",
    "Cholesky","Qr","Svd","SvdV2","MatrixInverse","MatrixDeterminant"
  ]

  static let arrayExact: Set<String> = [
    "Pack","Unpack","Concat","ConcatV2","Slice","StridedSlice","Split","SplitV",
    "Squeeze","ExpandDims","Pad","PadV2","Reshape","Transpose","Reverse","ReverseV2",
    "Tile","Shape","ShapeN","Rank","Size","Gather","GatherV2","GatherNd",
    "ScatterNd","Where","Unique","UniqueV2","TopK","TopKV2","OneHot"
  ]

  static let imageExact: Set<String> = [
    "ResizeBilinear","ResizeNearestNeighbor","ResizeBicubic","ResizeArea",
    "RGBToHSV","HSVToRGB","DecodeJpeg","DecodePng","EncodeJpeg","EncodePng"
  ]

  static let randomExact: Set<String> = [
    "RandomUniform","RandomUniformInt","RandomNormal","StatelessRandomNormal",
    "StatelessRandomUniform","StatelessRandomUniformInt","Multinomial",
    "TruncatedNormal","ParameterizedTruncatedNormal"
  ]

  static let controlExact: Set<String> = [
    "Switch","Merge","Enter","Exit","NextIteration","LoopCond","If","While","Case",
    "Identity","NoOp","StopGradient","PreventGradient"
  ]

  static let ioExact: Set<String> = [
    "ReadFile","WriteFile","RestoreV2","SaveV2","Restore","Save",
    "TFRecordReader","WholeFileReader","TextLineReader"
  ]

  static let mathExact: Set<String> = [
    // unary/binary common math
    "Add","AddV2","Sub","Mul","Div","RealDiv","Pow","SquaredDifference",
    "Maximum","Minimum","Mod","FloorMod",
    "Square","Sqrt","Rsqrt","Exp","Expm1","Log","Log1p",
    "Sin","Cos","Tan","Tanh","Asin","Acos","Atan","Sinh","Cosh",
    "Asinh","Acosh","Atanh","Erf","Erfc","Lgamma","Digamma",
    "Abs","Neg","Sign","Round","Rint","Ceil","Floor",
    "IsFinite","IsInf","IsNan",
    "Atan2","Equal","NotEqual","Less","LessEqual","Greater","GreaterEqual",
    "LogicalAnd","LogicalOr","LogicalNot",
    "Real","Imag","Complex","Conj","ComplexAbs"
  ]
}
