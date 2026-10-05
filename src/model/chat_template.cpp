#include "model/chat_template.hpp"

#include <string_view>

namespace ember {

namespace {

std::string_view lstrip_nl(std::string_view s) {
    while (!s.empty() && s.front() == '\n') s.remove_prefix(1);
    return s;
}
std::string_view rstrip_nl(std::string_view s) {
    while (!s.empty() && s.back() == '\n') s.remove_suffix(1);
    return s;
}
bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
bool ends_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(s.size() - p.size()) == p;
}

}  // namespace

std::string render_chat(const std::vector<ChatMessage> &messages, const ChatOptions &opt) {
    std::string out;
    const size_t n = messages.size();
    if (n > 0 && messages[0].role == "system") out += "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n";

    // The last real user query (tool responses travel as user messages).
    size_t last_query = n == 0 ? 0 : n - 1;
    for (size_t k = n; k-- > 0;) {
        const auto &m = messages[k];
        if (m.role == "user" && !(starts_with(m.content, "<tool_response>") && ends_with(m.content, "</tool_response>"))) {
            last_query = k;
            break;
        }
    }

    for (size_t i = 0; i < n; i++) {
        const ChatMessage &m = messages[i];
        if (m.role == "user" || (m.role == "system" && i != 0)) {
            out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            std::string_view content = m.content;
            std::string reasoning;
            size_t close = content.find("</think>");
            if (close != std::string_view::npos) {
                // reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                std::string_view before = rstrip_nl(content.substr(0, close));
                size_t open = before.rfind("<think>");
                if (open != std::string_view::npos) before = before.substr(open + 7);
                reasoning = std::string(lstrip_nl(before));
                // content = content.split('</think>')[-1].lstrip('\n')
                size_t last_close = content.rfind("</think>");
                content = lstrip_nl(content.substr(last_close + 8));
            }
            bool is_last = i + 1 == n;
            if (i > last_query && (is_last || !reasoning.empty())) {
                std::string_view r = lstrip_nl(rstrip_nl(reasoning));
                out += "<|im_start|>assistant\n<think>\n";
                out += r;
                out += "\n</think>\n\n";
                out += lstrip_nl(content);
            } else {
                out += "<|im_start|>assistant\n";
                out += content;
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (i == 0 || messages[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + m.content + "\n</tool_response>";
            if (i + 1 == n || messages[i + 1].role != "tool") out += "<|im_end|>\n";
        }
    }

    if (opt.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (!opt.enable_thinking) out += "<think>\n\n</think>\n\n";
    }
    return out;
}

}  // namespace ember
