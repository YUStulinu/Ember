#include "server/text_stream.hpp"

#include <algorithm>

namespace ember {

TextStream::TextStream(const Tokenizer &tok, std::vector<std::string> stops, bool reasoning_open)
    : think_dec_(tok), text_dec_(tok), stops_(std::move(stops)), thinking_(reasoning_open) {
    stops_.erase(std::remove(stops_.begin(), stops_.end(), std::string()), stops_.end());
    for (const auto &s : stops_) hold_ = std::max(hold_, s.size() - 1);
    if (auto id = tok.token_id("<think>")) think_open_ = *id;
    if (auto id = tok.token_id("</think>")) think_close_ = *id;
}

void TextStream::emit_text(const std::string &s_in, Piece &out) {
    if (stopped_) return;
    std::string s = s_in;
    if (!answer_started_) {
        // Qwen3 separates the reasoning block from the answer with blank lines.
        size_t first = s.find_first_not_of('\n');
        if (first == std::string::npos) return;
        s.erase(0, first);
        answer_started_ = true;
    }
    pending_ += s;
    // Earliest stop string in what we have.
    size_t best = std::string::npos;
    for (const auto &stop : stops_) {
        size_t at = pending_.find(stop);
        if (at != std::string::npos && at < best) {
            best = at;
            matched_ = stop;
        }
    }
    if (best != std::string::npos) {
        out.text += pending_.substr(0, best);
        pending_.clear();
        stopped_ = true;
        return;
    }
    // Keep the tail that could still grow into a stop string.
    if (pending_.size() > hold_) {
        size_t cut = pending_.size() - hold_;
        // Do not cut inside a UTF-8 character.
        while (cut > 0 && (static_cast<unsigned char>(pending_[cut]) & 0xC0) == 0x80) cut--;
        out.text += pending_.substr(0, cut);
        pending_.erase(0, cut);
    }
}

TextStream::Piece TextStream::push(const std::vector<int32_t> &tokens) {
    Piece out;
    for (int32_t t : tokens) {
        if (stopped_) break;
        if (t == think_open_) {
            thinking_ = true;
            continue;
        }
        if (t == think_close_) {
            std::string rest = think_dec_.flush();
            out.thinking += rest;
            thinking_ = false;
            continue;
        }
        if (thinking_) out.thinking += think_dec_.push(t);
        else emit_text(text_dec_.push(t), out);
    }
    return out;
}

TextStream::Piece TextStream::flush() {
    Piece out;
    out.thinking = think_dec_.flush();
    if (!stopped_) {
        emit_text(text_dec_.flush(), out);
        if (!stopped_) out.text += pending_;
    }
    pending_.clear();
    return out;
}

}  // namespace ember
