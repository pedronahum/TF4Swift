import Foundation
import CTensorFlow  // <-- bring TFE_OpSetAttr* and TF_DataType into scope

// MARK: - OpBuilder attribute setters (safe subset)
//
// Keep these setters isolated in this file to avoid accidental duplicates.
extension OpBuilder {
  @inlinable
  public func setAttr(_ name: String, int value: Int64) {
    name.withCString { cname in
      TFE_OpSetAttrInt(self.op, cname, value)
    }
  }

  @inlinable
  public func setAttr(_ name: String, ints values: [Int64]) {
    name.withCString { cname in
      values.withUnsafeBufferPointer { buf in
        TFE_OpSetAttrIntList(self.op, cname, buf.baseAddress, Int32(buf.count))
      }
    }
  }

  @inlinable
  public func setAttr(_ name: String, bool value: Bool) {
    name.withCString { cname in
      let v: UInt8 = value ? 1 : 0
      TFE_OpSetAttrBool(self.op, cname, v)
    }
  }

  @inlinable
  public func setAttr(_ name: String, string value: String) {
    name.withCString { cname in
      // Pass UTF‑8 bytes with explicit length (no intermediate Data allocation).
      let bytes = Array(value.utf8)
      bytes.withUnsafeBytes { raw in
        TFE_OpSetAttrString(self.op, cname, raw.baseAddress, raw.count)
      }
    }
  }

  @inlinable
  public func setAttrType(_ name: String, _ dtype: TF_DataType) {
    name.withCString { cname in
      TFE_OpSetAttrType(self.op, cname, dtype)
    }
  }
}
