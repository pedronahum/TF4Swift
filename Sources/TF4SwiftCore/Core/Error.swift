import CTensorFlow

public enum TensorFlowError: Error {
    case status(code: TF_Code, message: String)
}

