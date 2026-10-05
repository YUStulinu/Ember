// Turns generated tokens into text for an API response:
//   - splits Qwen3's reasoning (<think> ... </think>, single tokens) from the answer;
//   - never emits half a UTF-8 character;
//   - applies stop strings, holding back text that might be the start of one.
#pragma once

#include <string>
#include <vector>

#include "model/tokenizer.hpp"

namespace ember {

class TextStream {
public:
    TextStream(const Tokenizer &tok, std::vector<std::string> stops, bool reasoning_open);

    struct Piece {
        std::string thinking, text;
    };
    // Feeds tokens; returns the text that can be sent now.
    Piece push(const std::vector<int32_t> &tokens);
    // At the end: everything held back.
    Piece flush();

    bool stopped() const { return stopped_; }
    const std::string &stop_matched() const { return matched_; }

private:
    void emit_text(const std::string &s, Piece &out);

    StreamDecoder think_dec_, text_dec_;
    std::vector<std::string> stops_;
    size_t hold_ = 0;           // longest stop string - 1
    std::string pending_;        // answer text not yet sent (may start a stop string)
    bool thinking_;
    bool answer_started_ = false;
    bool stopped_ = false;
    std::string matched_;
    int think_open_ = -1, think_close_ = -1;
};

}  // namespace ember
