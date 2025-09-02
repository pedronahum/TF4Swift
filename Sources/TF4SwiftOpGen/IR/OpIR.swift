// MARK: - Generator IR

public struct OpIR: Hashable {
  public struct Input: Hashable {
    public var name: String
    public var typeAttr: String?   // e.g., "T"
    public var isList: Bool
  }
  public struct Output: Hashable {
    public var name: String
    public var typeAttr: String?
    public var isList: Bool
  }
  public enum AttrType { case bool, int, float, type, shape, string, list }
  public struct Attr: Hashable {
    public var name: String
    public var type: AttrType
    public var hasDefault: Bool
    public var defaultString: String? // preformatted default for codegen
  }
  public struct Endpoint: Hashable {
    /// Canonical or alias endpoint like "math.add" or "nn.relu".
    public var fqName: String
  }
  public enum Group: String, Hashable { case math, nn, array, linalg, image, random, io, control, other }

  public var opName: String         // Graph op name, e.g. "AddV2"
  public var inputs: [Input]
  public var outputs: [Output]
  public var attrs: [Attr]
  public var endpoints: [Endpoint]  // possibly multiple
  public var group: Group           // chosen from endpoints or heuristics
}
