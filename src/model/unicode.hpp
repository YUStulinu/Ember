// The Unicode pieces the tokenizer needs: UTF-8 decoding, the character
// classes of the pre-tokenizer regex, and NFC normalization.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ember::unicode {

constexpr uint32_t kReplacement = 0xFFFD;

// Decodes the code point at s[i] and advances i. Invalid or truncated
// sequences decode as U+FFFD and advance by one byte.
uint32_t next_cp(std::string_view s, size_t &i);
void append_utf8(std::string &out, uint32_t cp);

bool is_letter(uint32_t cp);  // \p{L}
bool is_number(uint32_t cp);  // \p{N}
bool is_space(uint32_t cp);   // \s (White_Space)
int combining_class(uint32_t cp);

// NFC normalization. Invalid UTF-8 becomes U+FFFD.
std::string nfc(std::string_view s);

// Replaces each maximal invalid subsequence with U+FFFD (the WHATWG / Rust
// from_utf8_lossy rule, so decoded text matches the reference tokenizer).
std::string to_valid_utf8(std::string_view s);

// Length of the longest prefix of s that does not end inside a UTF-8 sequence
// (used to stream text without splitting a character across two chunks).
size_t complete_utf8_prefix(std::string_view s);

const char *data_version();

}  // namespace ember::unicode
