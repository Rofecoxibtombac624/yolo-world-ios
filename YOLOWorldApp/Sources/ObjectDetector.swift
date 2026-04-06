import AVFoundation
import CoreML
import UIKit

struct Detection {
    let box: CGRect      // normalised 0-1, origin top-left
    let score: Float
}

final class ObjectDetector: NSObject, ObservableObject {
    @Published var activeQuery: String = "person"
    @Published var activeAction: DetectionAction = .track
    @Published var matchInfo: String = "Tracking: person"
    @Published var detections: [Detection] = []
    @Published var distanceMeters: Float?

    let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private let outputSync: AVCaptureDataOutputSynchronizer?
    private let queue = DispatchQueue(label: "detection", qos: .userInitiated)
    private let ciContext = CIContext()
    private var isProcessing = false

    private let detectorModel: MLModel
    private let textModel: MLModel
    private let tokenizer: CLIPTokenizer

    // Cached text embedding — updated when user types a new word
    private var currentEmbedding: MLMultiArray

    // Latest depth map from LiDAR
    private var latestDepthMap: CVPixelBuffer?

    private let confidenceThreshold: Float = 0.15
    private let iouThreshold: Float = 0.45

    override init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        detectorModel = try! YOLOWorldDetector(configuration: config).model
        textModel = try! YOLOWorldText(configuration: config).model
        tokenizer = CLIPTokenizer()

        // Encode default query "person"
        currentEmbedding = Self.encodeQuery("person", tokenizer: tokenizer, textModel: textModel)

        // Try to set up synchronized output if depth is available
        outputSync = nil

        super.init()
        setupCamera()
    }

    private static func encodeQuery(_ query: String, tokenizer: CLIPTokenizer, textModel: MLModel) -> MLMultiArray {
        let tokens = tokenizer.tokenize(query)
        let tokenArray = try! MLMultiArray(shape: [1, 77], dataType: .int32)
        for (i, t) in tokens.enumerated() { tokenArray[i] = NSNumber(value: t) }

        let input = try! MLDictionaryFeatureProvider(dictionary: ["tokens": tokenArray])
        let output = try! textModel.prediction(from: input)
        let embedding = output.featureValue(for: "embedding")!.multiArrayValue!

        // Reshape [1, 512] → [1, 1, 512]
        let reshaped = try! MLMultiArray(shape: [1, 1, 512], dataType: .float16)
        for i in 0..<512 { reshaped[[0, 0, i] as [NSNumber]] = embedding[[0, i] as [NSNumber]] }
        return reshaped
    }

    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .photo

        // Prefer LiDAR camera, fall back to dual wide, then wide angle
        let device: AVCaptureDevice? =
            AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)
            ?? AVCaptureDevice.default(.builtInDualWideCamera, for: .video, position: .back)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)

        guard let device, let input = try? AVCaptureDeviceInput(device: device) else {
            print("Camera not available")
            captureSession.commitConfiguration()
            return
        }

        if captureSession.canAddInput(input) { captureSession.addInput(input) }

        // Video output
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }

        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90
        }

        // Depth output (LiDAR)
        if captureSession.canAddOutput(depthOutput) {
            captureSession.addOutput(depthOutput)
            depthOutput.isFilteringEnabled = true
            depthOutput.setDelegate(self, callbackQueue: queue)
            if let connection = depthOutput.connection(with: .depthData) {
                connection.videoRotationAngle = 90
            }
        }

        captureSession.commitConfiguration()
    }

    func startSession() {
        guard !captureSession.isRunning else { return }
        queue.async { [weak self] in self?.captureSession.startRunning() }
    }

    func stopSession() {
        guard captureSession.isRunning else { return }
        queue.async { [weak self] in self?.captureSession.stopRunning() }
    }

    func setQuery(_ query: String, action: DetectionAction = .track) {
        let trimmed = query.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let embedding = Self.encodeQuery(trimmed, tokenizer: self.tokenizer, textModel: self.textModel)
            DispatchQueue.main.async {
                self.currentEmbedding = embedding
                self.activeQuery = trimmed
                self.activeAction = action
                switch action {
                case .track:    self.matchInfo = "Tracking: \(trimmed)"
                case .distance: self.matchInfo = "Measuring: \(trimmed)"
                case .count:    self.matchInfo = "Counting: \(trimmed)"
                }
            }
        }
    }

    private func runInference(on pixelBuffer: CVPixelBuffer) {
        let embedding = currentEmbedding

        do {
            let imageFeature = try MLFeatureValue(pixelBuffer: pixelBuffer)
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image": imageFeature,
                "txt_feats": MLFeatureValue(multiArray: embedding)
            ])
            let output = try detectorModel.prediction(from: input)
            let raw = output.featureValue(for: "detections")!.multiArrayValue!

            let parsed = parseDetections(raw)
            let nmsed = nms(parsed)

            // Sample depth at center of top detection
            let distance = sampleDepth(for: nmsed.first)

            DispatchQueue.main.async {
                self.detections = nmsed
                self.distanceMeters = distance
                self.isProcessing = false
            }
        } catch {
            print("Inference error: \(error)")
            DispatchQueue.main.async { self.isProcessing = false }
        }
    }

    private func sampleDepth(for detection: Detection?) -> Float? {
        guard let detection, let depthMap = latestDepthMap else { return nil }

        let depthWidth = CVPixelBufferGetWidth(depthMap)
        let depthHeight = CVPixelBufferGetHeight(depthMap)

        // Detection box center (normalised 0-1) → depth pixel coords
        let cx = Int(detection.box.midX * CGFloat(depthWidth))
        let cy = Int(detection.box.midY * CGFloat(depthHeight))

        let clampedX = max(0, min(depthWidth - 1, cx))
        let clampedY = max(0, min(depthHeight - 1, cy))

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        let pixelFormat = CVPixelBufferGetPixelFormatType(depthMap)

        if pixelFormat == kCVPixelFormatType_DepthFloat32 {
            let ptr = baseAddress.assumingMemoryBound(to: Float.self)
            let depth = ptr[clampedY * (bytesPerRow / MemoryLayout<Float>.size) + clampedX]
            return depth > 0 ? depth : nil
        } else if pixelFormat == kCVPixelFormatType_DisparityFloat16 || pixelFormat == kCVPixelFormatType_DepthFloat16 {
            let ptr = baseAddress.assumingMemoryBound(to: Float16.self)
            let depth = Float(ptr[clampedY * (bytesPerRow / MemoryLayout<Float16>.size) + clampedX])
            return depth > 0 ? depth : nil
        }

        return nil
    }

    /// Parse raw [1, 5, 8400] output into Detection array
    private func parseDetections(_ raw: MLMultiArray) -> [Detection] {
        let numAnchors = raw.shape[2].intValue  // 8400
        var results: [Detection] = []

        for j in 0..<numAnchors {
            let score = raw[[0, 4, j] as [NSNumber]].floatValue
            guard score > confidenceThreshold else { continue }

            let cx = raw[[0, 0, j] as [NSNumber]].floatValue
            let cy = raw[[0, 1, j] as [NSNumber]].floatValue
            let w  = raw[[0, 2, j] as [NSNumber]].floatValue
            let h  = raw[[0, 3, j] as [NSNumber]].floatValue

            // cxcywh in pixel coords → normalised 0-1, origin top-left
            let x = CGFloat((cx - w / 2) / 640)
            let y = CGFloat((cy - h / 2) / 640)
            let bw = CGFloat(w / 640)
            let bh = CGFloat(h / 640)

            let box = CGRect(x: max(0, x), y: max(0, y),
                             width: min(1 - max(0, x), bw),
                             height: min(1 - max(0, y), bh))
            results.append(Detection(box: box, score: score))
        }

        return results.sorted { $0.score > $1.score }
    }

    /// Non-maximum suppression
    private func nms(_ detections: [Detection]) -> [Detection] {
        var kept: [Detection] = []
        var suppressed = Set<Int>()

        for i in 0..<detections.count {
            guard !suppressed.contains(i) else { continue }
            kept.append(detections[i])

            for j in (i + 1)..<detections.count {
                guard !suppressed.contains(j) else { continue }
                if iou(detections[i].box, detections[j].box) > iouThreshold {
                    suppressed.insert(j)
                }
            }
        }
        return kept
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }
        let interArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return Float(interArea / unionArea)
    }
}

extension ObjectDetector: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard !isProcessing,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        isProcessing = true
        let resized = resizePixelBuffer(pixelBuffer, to: CGSize(width: 640, height: 640))
        runInference(on: resized)
    }

    private func resizePixelBuffer(_ buffer: CVPixelBuffer, to size: CGSize) -> CVPixelBuffer {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let imageSize = CGSize(
            width: CVPixelBufferGetWidth(buffer),
            height: CVPixelBufferGetHeight(buffer)
        )
        let scale = CGAffineTransform(
            scaleX: size.width / imageSize.width,
            y: size.height / imageSize.height
        )

        var resized: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width), Int(size.height),
            kCVPixelFormatType_32BGRA,
            nil,
            &resized
        )

        ciContext.render(ciImage.transformed(by: scale), to: resized!)
        return resized!
    }
}

extension ObjectDetector: AVCaptureDepthDataOutputDelegate {
    func depthDataOutput(_ output: AVCaptureDepthDataOutput,
                         didOutput depthData: AVDepthData,
                         timestamp: CMTime,
                         connection: AVCaptureConnection) {
        // Convert to float32 depth map if needed
        let converted = depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depthData
            : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        latestDepthMap = converted.depthDataMap
    }
}
