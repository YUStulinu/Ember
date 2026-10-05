// A byte-level BPE tokenizer compatible with Hugging Face tokenizer.json files
// of the GPT-2 / Qwen family:
//
//   text --split on added tokens--> segments
//        --NFC--> --pre-tokenizer regex--> pieces
//        --bytes, mapped to the byte-level alphabet--> --BPE merges--> ids
//
// The pre-tokenizer regex (Qwen2/Qwen3) is matched by hand-written code rather
// than a regex engine: it is ~20x faster and has no dependency, and the tests
// check it against the official tokenizer on hard inputs.
#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ember {

class Tokenizer {
public:
    static Tokenizer load(const std::string &tokenizer_json_path);

    Tokenizer() = default;
    Tokenizer(Tokenizer &&other) noexcept;
    Tokenizer &operator=(Tokenizer &&other) noexcept;

    // parse_special: recognize added tokens such as <|im_start|> in the text.
    // Thread-safe.
    std::vector<int> encode(std::string_view text, bool parse_special = true) const;

    // Concatenated token bytes as valid UTF-8 (invalid sequences become U+FFFD).
    std::string decode(std::span<const int> ids, bool skip_special = false) const;
    // The raw bytes of one token (may be part of a UTF-8 character).
    const std::string &token_bytes(int id) const;

    int vocab_size() const { return static_cast<int>(id_to_bytes_.size()); }
    std::optional<int> token_id(std::string_view token) const;  // exact token text, e.g. "<|im_end|>"
    bool is_special(int id) const;
    bool is_added(int id) const;

    // The pre-tokenizer on its own (exposed for tests).
    static void pre_tokenize(std::string_view text, std::vector<std::string_view> &pieces);

private:
    struct Merge {
        int rank;
        int result;
    };
    struct Added {
        std::string text;
        int id;
        bool special;
    };

    void encode_segment(std::string_view text, std::vector<int> &out) const;
    void bpe(std::string_view piece, std::vector<int> &out) const;

    std::vector<std::string> id_to_bytes_;            // raw bytes of every token
    std::unordered_map<std::string, int> bytes_to_id_;
    std::unordered_map<uint64_t, Merge> merges_;      // (left id, right id) -> merge
    int byte_id_[256] = {};                           // token of each single byte
    std::vector<Added> added_;                        // longest first
    std::vector<uint8_t> flags_;                      // per id: 1 = added, 2 = special

    mutable std::mutex cache_mutex_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

// Turns a stream of token ids into text without ever emitting half a UTF-8
// character: bytes of an incomplete character are held back until the next
// token completes it.
class StreamDecoder {
public:
    explicit StreamDecoder(const Tokenizer &tok, bool skip_special = true) : tok_(&tok), skip_special_(skip_special) {}
    std::string push(int id);  // newly completed text (possibly empty)
    std::string flush();       // whatever is left, made valid UTF-8

private:
    const Tokenizer *tok_;
    bool skip_special_;
    std::string pending_;
};

}  // namespace ember
