import Foundation

struct CLIPTokenizer {
    let vocab: [String: Int]
    let merges: [(String, String)]
    let sotToken = 49406
    let eotToken = 49407
    let maxLength = 77

    init() {
        let url = Bundle.main.url(forResource: "clip_tokenizer", withExtension: "json")!
        let data = try! Data(contentsOf: url)
        let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]
        self.vocab = json["vocab"] as! [String: Int]
        let mergeStrings = json["merges"] as! [String]
        self.merges = mergeStrings.map {
            let parts = $0.split(separator: " ", maxSplits: 1).map(String.init)
            return (parts[0], parts[1])
        }
    }

    func tokenize(_ text: String) -> [Int32] {
        let lower = text.lowercased().trimmingCharacters(in: .whitespaces)
        var tokens = [sotToken]
        // Split into words, tokenize each
        let words = lower.split(separator: " ").map(String.init)
        for word in words {
            var chars = word.map { String($0) }
            guard !chars.isEmpty else { continue }
            chars[chars.count - 1] += "</w>"
            var pieces = chars
            var pairs = getBigrams(pieces)
            while !pairs.isEmpty {
                guard let best = pairs.min(by: {
                    (mergeRank($0) ?? .max) < (mergeRank($1) ?? .max)
                }), mergeRank(best) != nil else { break }
                pieces = applyMerge(pieces, pair: best)
                pairs = getBigrams(pieces)
            }
            for piece in pieces {
                if let id = vocab[piece] { tokens.append(id) }
            }
        }
        tokens.append(eotToken)
        if tokens.count > maxLength {
            tokens = Array(tokens.prefix(maxLength - 1)) + [eotToken]
        }
        while tokens.count < maxLength { tokens.append(0) }
        return tokens.map(Int32.init)
    }

    private func mergeRank(_ pair: (String, String)) -> Int? {
        merges.firstIndex(where: { $0 == pair })
    }

    private func getBigrams(_ word: [String]) -> [(String, String)] {
        guard word.count > 1 else { return [] }
        return zip(word, word.dropFirst()).map { ($0, $1) }
    }

    private func applyMerge(_ word: [String], pair: (String, String)) -> [String] {
        var result: [String] = []
        var i = 0
        while i < word.count {
            if i < word.count - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                result.append(pair.0 + pair.1)
                i += 2
            } else {
                result.append(word[i])
                i += 1
            }
        }
        return result
    }
}
