import SwiftUI

struct ContentView: View {
    @StateObject private var detector = ObjectDetector()
    @State private var searchText = ""
    @State private var isParsing = false
    @State private var isModelLoading = true
    @State private var hint: String?
    @FocusState private var isFocused: Bool

    private let parser = IntentParser()

    var body: some View {
        ZStack {
            CameraPreviewView(detector: detector)
                .ignoresSafeArea()
                .onTapGesture { isFocused = false }

            DetectionOverlay(detections: detector.detections)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            VStack {
                // Top display — depends on active action
                switch detector.activeAction {
                case .distance:
                    if let distance = detector.distanceMeters {
                        Text(formatDistance(distance))
                            .font(.system(size: 72, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                            .shadow(color: .black.opacity(0.6), radius: 8, x: 0, y: 2)
                            .padding(.top, 60)
                            .transition(.opacity)
                    }
                case .count:
                    Text("\(detector.detections.count)")
                        .font(.system(size: 72, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                        .shadow(color: .black.opacity(0.6), radius: 8, x: 0, y: 2)
                        .padding(.top, 60)
                        .transition(.opacity)
                case .track:
                    EmptyView()
                }

                Spacer()

                if isModelLoading {
                    HStack(spacing: 8) {
                        ProgressView()
                            .tint(.white)
                        Text("Loading language model...")
                            .font(.subheadline)
                            .foregroundStyle(.white)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(.black.opacity(0.6), in: Capsule())
                    .padding(.bottom, 8)
                }

                if let hint {
                    Text(hint)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.8))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.black.opacity(0.5), in: Capsule())
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                        .padding(.bottom, 4)
                }

                Text(detector.matchInfo)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.black.opacity(0.5), in: Capsule())
                    .padding(.bottom, 8)

                HStack(spacing: 0) {
                    TextField(
                        isModelLoading
                            ? "Loading model..."
                            : "\"How far is that person?\"",
                        text: $searchText
                    )
                    .focused($isFocused)
                    .textInputAutocapitalization(.never)
                    .disabled(isModelLoading)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                    .onSubmit { submitQuery() }

                    if isParsing {
                        ProgressView()
                            .padding(.trailing, 14)
                    } else {
                        Button {
                            submitQuery()
                        } label: {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.system(size: 30))
                                .foregroundStyle(
                                    canSubmit ? .white : .gray.opacity(0.4)
                                )
                        }
                        .disabled(!canSubmit)
                        .padding(.trailing, 8)
                    }
                }
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 24))
                .overlay(
                    RoundedRectangle(cornerRadius: 24)
                        .stroke(.white.opacity(0.15), lineWidth: 1)
                )
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            }
            .animation(.default, value: hint)
            .animation(.default, value: isModelLoading)
            .animation(.easeOut(duration: 0.15), value: detector.distanceMeters)
            .animation(.easeOut(duration: 0.15), value: detector.detections.count)
        }
        .task {
            await parser.load()
            isModelLoading = false
        }
    }

    private var canSubmit: Bool {
        !isModelLoading && !isParsing
            && !searchText.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private func formatDistance(_ meters: Float) -> String {
        if meters < 1 {
            return String(format: "%.0fcm", meters * 100)
        } else {
            return String(format: "%.1fm", meters)
        }
    }

    private func submitQuery() {
        let query = searchText.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty, !isParsing, !isModelLoading else { return }

        searchText = ""
        isFocused = false
        isParsing = true
        hint = nil

        Task {
            if let intent = await parser.parse(query) {
                detector.setQuery(intent.class, action: intent.action)
            } else {
                hint = "Try describing an object to track"
            }
            isParsing = false
        }
    }
}
