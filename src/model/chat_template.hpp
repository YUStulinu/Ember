// Turns a conversation into the prompt text the model was trained on.
//
// Qwen3 uses ChatML (<|im_start|>role\n...<|im_end|>) plus rules about
// <think> blocks; this reproduces the official Jinja template (without tool
// definitions) exactly, and the tests compare it with transformers' output.
#pragma once

#include <string>
#include <vector>

namespace ember {

struct ChatMessage {
    std::string role;     // "system", "user", "assistant" or "tool"
    std::string content;
};

struct ChatOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = false;  // Qwen3: false adds an empty <think></think> block
};

std::string render_chat(const std::vector<ChatMessage> &messages, const ChatOptions &options);

}  // namespace ember
