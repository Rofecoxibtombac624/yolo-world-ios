import LLM
import Foundation

actor IntentParser {

    private var modelURL: URL?
    private(set) var isLoaded = false

    private static let remoteURL = URL(string: "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf")!
    private static let modelFilename = "Qwen3-0.6B-Q4_K_M.gguf"

    /// Call once at app start — downloads model on first run (~400MB)
    func load() async {
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let localURL = docsDir.appendingPathComponent(Self.modelFilename)

        if !FileManager.default.fileExists(atPath: localURL.path) {
            do {
                let (tempURL, _) = try await URLSession.shared.download(from: Self.remoteURL)
                try FileManager.default.moveItem(at: tempURL, to: localURL)
            } catch {
                print("Model download failed: \(error)")
                isLoaded = true
                return
            }
        }

        // Verify model loads successfully
        if LLM(from: localURL, template: .chatML("test"), temp: 0.1, maxTokenCount: 512) != nil {
            modelURL = localURL
        }
        isLoaded = true
    }

    /// Call when user submits text. Returns nil if input is unparseable.
    func parse(_ userInput: String) async -> DetectionIntent? {
        guard let modelURL else { return nil }

        // Fresh LLM instance each call — avoids KV cache state issues
        guard let bot = LLM(
            from: modelURL,
            template: .chatML(systemPrompt),
            temp: 0.1,
            maxTokenCount: 512
        ) else { return nil }

        let prompt = "/no_think\nExtract the object from: \"\(userInput)\"\nRespond with JSON only: {\"class\": \"<noun>\", \"action\": \"distance\"}"

        let formatted = bot.preprocess(prompt, [])
        let output = await bot.getCompletion(from: formatted)

        print("[IntentParser] raw output: \(output)")

        // Extract JSON from response
        guard let jsonStart = output.firstIndex(of: "{"),
              let jsonEnd = output.lastIndex(of: "}") else {
            print("[IntentParser] no JSON found in output")
            return nil
        }

        let jsonString = String(output[jsonStart...jsonEnd])
        guard let data = jsonString.data(using: .utf8),
              let intent = try? JSONDecoder().decode(DetectionIntent.self, from: data) else {
            print("[IntentParser] failed to decode: \(jsonString)")
            return nil
        }

        return intent
    }

    private let systemPrompt = """
    You extract object detection intent from natural language.
    Always respond with valid JSON only, no other text.
    The "class" field must be a single physical noun in lowercase.
    The "action" field must be one of: "track", "distance", "count".
    - "distance" = user wants to measure how far something is
    - "count" = user wants to count how many of something there are
    - "track" = user just wants to detect/find/see something (default)
    Examples:
    Input: "how far is that person" → {"class": "person", "action": "distance"}
    Input: "distance to the chair" → {"class": "chair", "action": "distance"}
    Input: "count the bottles" → {"class": "bottle", "action": "count"}
    Input: "how many chairs" → {"class": "chair", "action": "count"}
    Input: "find dogs" → {"class": "dog", "action": "track"}
    Input: "detect laptops" → {"class": "laptop", "action": "track"}
    Input: "show me cups" → {"class": "cup", "action": "track"}
    """
}
