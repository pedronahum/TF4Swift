import Testing
@testable import TF4SwiftOpGen

@Suite("OpGen golden tests")
struct GenGoldenTests {

    @Test("MatMul has defaulted transpose flags")
    func matmul_has_defaulted_flags() throws {
        // Minimal MatMul op_def with defaulted transpose flags.
        let pb = """
        op_def {
          name: "MatMul"
          input_arg { name: "a" type_attr: "T" }
          input_arg { name: "b" type_attr: "T" }
          output_arg { name: "product" type_attr: "T" }
          attr { name: "T" type: "type" }
          attr { name: "transpose_a" type: "bool" default_value { b: false } }
          attr { name: "transpose_b" type: "bool" default_value { b: false } }
        }
        """

        // Parse the op from the provided pbtxt.
        guard let op = findOp(named: "MatMul", in: pb) else {
            Issue.record("failed to parse MatMul op")
            return
        }

        // Ask the emitter for the Swift wrapper text.
        let (domain, fileName, contents, _) = try emitSwiftForOp(op, in: pb)

        // Domain & filename are part of the acceptance contract.
        #expect(domain == "Math")
        #expect(fileName == "MatMul.swift")

        // Surface defaulted parameters.
        #expect(contents.contains("transposeA: Bool = false"))
        #expect(contents.contains("transposeB: Bool = false"))

        // The wrapper must call through the builder with the MatMul op name.
        // Note: we look for `build("MatMul")` (no leading dot) because build is the
        // first call in the chain, not a method off a previous expression.
        #expect(
            contents.contains(#"build("MatMul")"#),
            // (Optional) sanity check note for future regressions.
            "The wrapper must call through the builder with the MatMul op name."
        )

        // Attribute wiring must pass the booleans positionally (no extraneous labels).
        #expect(contents.contains(#".attr("transpose_a", transposeA)"#))
        #expect(contents.contains(#".attr("transpose_b", transposeB)"#))
    }
}
