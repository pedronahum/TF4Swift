import Foundation

public enum BroadcastError: Error, CustomStringConvertible, Sendable {
    case incompatible(lhs: [Int64], rhs: [Int64])
    public var description: String {
        switch self {
        case let .incompatible(lhs, rhs):
            return "Cannot broadcast shapes \(lhs) and \(rhs)."
        }
    }
}

/// Numpy/TF-style broadcasting:
/// Align shapes from the trailing dims; each pair must be equal or one must be 1.
public func broadcastedShape(_ a: [Int64], _ b: [Int64]) throws -> [Int64] {
    let na = a.count, nb = b.count, n = max(na, nb)
    var out: [Int64] = []
    out.reserveCapacity(n)

    for i in 0..<n {
        let da = i < na ? a[na - 1 - i] : 1
        let db = i < nb ? b[nb - 1 - i] : 1
        if da == db || da == 1 || db == 1 {
            out.append(max(da, db))
        } else {
            throw BroadcastError.incompatible(lhs: a, rhs: b)
        }
    }
    return out.reversed()
}
