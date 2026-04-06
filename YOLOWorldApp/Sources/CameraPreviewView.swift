import AVFoundation
import SwiftUI

struct CameraPreviewView: UIViewRepresentable {
    @ObservedObject var detector: ObjectDetector

    func makeUIView(context: Context) -> UIView {
        let view = CameraUIView()
        view.previewLayer.session = detector.captureSession
        detector.startSession()
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}

    static func dismantleUIView(_ uiView: UIView, coordinator: ()) {
        // Session stopped when detector is deallocated
    }
}

private class CameraUIView: UIView {
    let previewLayer = AVCaptureVideoPreviewLayer()

    override init(frame: CGRect) {
        super.init(frame: frame)
        previewLayer.videoGravity = .resizeAspectFill
        layer.addSublayer(previewLayer)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
    }
}
