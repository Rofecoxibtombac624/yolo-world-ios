import Foundation

enum DetectionAction: String, Codable {
    case track
    case distance
    case count
}

struct DetectionIntent: Codable {
    let `class`: String
    let action: DetectionAction
}
