import SwiftUI

struct DetectionOverlay: View {
    let detections: [Detection]

    var body: some View {
        GeometryReader { geo in
            ForEach(Array(detections.enumerated()), id: \.offset) { _, det in
                let rect = CGRect(
                    x: det.box.minX * geo.size.width,
                    y: det.box.minY * geo.size.height,
                    width: det.box.width * geo.size.width,
                    height: det.box.height * geo.size.height
                )
                let pct = Int(det.score * 100)

                ZStack(alignment: .topLeading) {
                    Rectangle()
                        .stroke(Color.green, lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)

                    Text("\(pct)%")
                        .font(.caption2.bold())
                        .foregroundColor(.black)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                        .background(Color.green)
                        .offset(y: -20)
                }
                .position(x: rect.midX, y: rect.midY)
            }
        }
    }
}
