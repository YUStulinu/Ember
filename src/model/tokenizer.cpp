#include "model/tokenizer.hpp"

#include <algorithm>
#include <queue>

#include "common/error.hpp"
#include "common/json.hpp"
#include "common/mmap_file.hpp"
#include "model/unicode.hpp"

namespace ember {

namespace {

// GPT-2's byte-level alphabet: every byte maps to a printable code point, so
// token strings in tokenizer.json are valid text. Printable bytes map to
// themselves, the other 68 to U+0100 and up.
struct ByteAlphabet {
    uint32_t byte_to_cp[256];
    int cp_to_byte[324];  // inverse, -1 if not in the alphabet
    ByteAlphabet() {
        std::fill(std::begin(cp_to_byte), std::end(cp_to_byte), -1);
        bool direct[256] = {};
        for (int b = '!'; b <= '~'; b++) direct[b] = true;
        for (int b = 0xA1; b <= 0xAC; b++) direct[b] = true;
        for (int b = 0xAE; b <= 0xFF; b++) direct[b] = true;
        uint32_t next = 256;
        for (int b = 0; b < 256; b++) byte_to_cp[b] = direct[b] ? static_cast<uint32_t>(b) : next++;
        for (int b = 0; b < 256; b++) cp_to_byte[byte_to_cp[b]] = b;
    }
};

const ByteAlphabet &alphabet() {
    static const ByteAlphabet a;
    return a;
}

// "Ġhello" (byte-level alphabet) -> " hello" (raw bytes).
std::string from_byte_level(std::string_view s, bool *ok) {
    std::string out;
    out.reserve(s.size());
    *ok = true;
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = unicode::next_cp(s, i);
        if (cp >= 324 || alphabet().cp_to_byte[cp] < 0) {
            *ok = false;
            return {};
        }
        out += static_cast<char>(alphabet().cp_to_byte[cp]);
    }
    return out;
}

uint64_t pair_key(int a, int b) { return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b); }

constexpr size_t kCacheLimit = 1 << 17;

}  // namespace

Tokenizer::Tokenizer(Tokenizer &&o) noexcept { *this = std::move(o); }

Tokenizer &Tokenizer::operator=(Tokenizer &&o) noexcept {
    if (this != &o) {
        id_to_bytes_ = std::move(o.id_to_bytes_);
        bytes_to_id_ = std::move(o.bytes_to_id_);
        merges_ = std::move(o.merges_);
        std::copy(std::begin(o.byte_id_), std::end(o.byte_id_), std::begin(byte_id_));
        added_ = std::move(o.added_);
        flags_ = std::move(o.flags_);
        cache_.clear();
    }
    return *this;
}

Tokenizer Tokenizer::load(const std::string &path) {
    json::Value root = json::Value::parse(read_file(path));
    const json::Value &model = root["model"];
    if (model.get_string("type", "") != "BPE") fail("{}: only BPE tokenizers are supported", path);
    if (const json::Value *norm = root.find("normalizer"); norm && !norm->is_null() && norm->get_string("type", "") != "NFC")
        fail("{}: unsupported normalizer {}", path, norm->get_string("type", "?"));

    Tokenizer t;
    const json::Object &vocab = model["vocab"].as_object();
    int max_id = -1;
    for (const auto &[_, id] : vocab) max_id = std::max(max_id, static_cast<int>(id.as_int()));
    for (const auto &a : root["added_tokens"].as_array()) max_id = std::max(max_id, static_cast<int>(a["id"].as_int()));
    t.id_to_bytes_.resize(static_cast<size_t>(max_id) + 1);
    t.flags_.assign(static_cast<size_t>(max_id) + 1, 0);
    t.bytes_to_id_.reserve(vocab.size() * 2);

    for (const auto &[text, idv] : vocab) {
        int id = static_cast<int>(idv.as_int());
        bool ok;
        std::string bytes = from_byte_level(text, &ok);
        if (!ok) fail("{}: vocabulary entry {} is not in the byte-level alphabet", path, id);
        t.bytes_to_id_.emplace(bytes, id);
        t.id_to_bytes_[static_cast<size_t>(id)] = std::move(bytes);
    }
    for (int b = 0; b < 256; b++) {
        auto it = t.bytes_to_id_.find(std::string(1, static_cast<char>(b)));
        if (it == t.bytes_to_id_.end()) fail("{}: byte 0x{:02X} has no token", path, b);
        t.byte_id_[b] = it->second;
    }

    const json::Array &merges = model["merges"].as_array();
    t.merges_.reserve(merges.size() * 2);
    int rank = 0;
    for (const auto &m : merges) {
        std::string left, right;
        if (m.is_array()) {
            left = m[0].as_string();
            right = m[1].as_string();
        } else {  // older format: "left right"
            const std::string &s = m.as_string();
            size_t sp = s.find(' ');
            if (sp == std::string::npos) fail("{}: malformed merge \"{}\"", path, s);
            left = s.substr(0, sp);
            right = s.substr(sp + 1);
        }
        bool ok1, ok2;
        std::string lb = from_byte_level(left, &ok1), rb = from_byte_level(right, &ok2);
        auto li = t.bytes_to_id_.find(lb), ri = t.bytes_to_id_.find(rb), mi = t.bytes_to_id_.find(lb + rb);
        if (!ok1 || !ok2 || li == t.bytes_to_id_.end() || ri == t.bytes_to_id_.end() || mi == t.bytes_to_id_.end())
            fail("{}: merge {} refers to unknown tokens", path, rank);
        t.merges_.emplace(pair_key(li->second, ri->second), Merge{rank, mi->second});
        rank++;
    }

    for (const auto &a : root["added_tokens"].as_array()) {
        int id = static_cast<int>(a["id"].as_int());
        bool special = a.get_bool("special", false);
        std::string text = a["content"].as_string();
        t.id_to_bytes_[static_cast<size_t>(id)] = text;
        t.flags_[static_cast<size_t>(id)] = special ? 3 : 1;
        t.added_.push_back({text, id, special});
    }
    std::sort(t.added_.begin(), t.added_.end(), [](const Added &a, const Added &b) { return a.text.size() > b.text.size(); });
    return t;
}

const std::string &Tokenizer::token_bytes(int id) const {
    static const std::string empty;
    if (id < 0 || id >= vocab_size()) return empty;
    return id_to_bytes_[static_cast<size_t>(id)];
}

std::optional<int> Tokenizer::token_id(std::string_view token) const {
    for (const auto &a : added_)
        if (a.text == token) return a.id;
    auto it = bytes_to_id_.find(std::string(token));
    if (it != bytes_to_id_.end()) return it->second;
    return std::nullopt;
}

bool Tokenizer::is_special(int id) const { return id >= 0 && id < vocab_size() && (flags_[static_cast<size_t>(id)] & 2); }
bool Tokenizer::is_added(int id) const { return id >= 0 && id < vocab_size() && (flags_[static_cast<size_t>(id)] & 1); }

// --- the pre-tokenizer -------------------------------------------------------
//
// Hand-written matcher for the Qwen2/Qwen3 split regex, alternatives tried in order:
//   1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   2. [^\r\n\p{L}\p{N}]?\p{L}+
//   3. \p{N}
//   4.  ?[^\s\p{L}\p{N}]+[\r\n]*
//   5. \s*[\r\n]+
//   6. \s+(?!\S)
//   7. \s+
void Tokenizer::pre_tokenize(std::string_view s, std::vector<std::string_view> &pieces) {
    // Decode once: code points and their byte offsets (offs has one extra entry: the end).
    std::vector<uint32_t> cps;
    std::vector<uint32_t> offs;
    cps.reserve(s.size());
    offs.reserve(s.size() + 1);
    for (size_t i = 0; i < s.size();) {
        offs.push_back(static_cast<uint32_t>(i));
        cps.push_back(unicode::next_cp(s, i));
    }
    offs.push_back(static_cast<uint32_t>(s.size()));
    const size_t n = cps.size();

    auto L = [&](size_t k) { return k < n && unicode::is_letter(cps[k]); };
    auto N = [&](size_t k) { return k < n && unicode::is_number(cps[k]); };
    auto S = [&](size_t k) { return k < n && unicode::is_space(cps[k]); };
    auto NL = [&](size_t k) { return k < n && (cps[k] == '\r' || cps[k] == '\n'); };
    auto other = [&](size_t k) { return k < n && !unicode::is_space(cps[k]) && !unicode::is_letter(cps[k]) && !unicode::is_number(cps[k]); };
    auto lower = [&](size_t k) -> uint32_t {
        if (k >= n) return 0;
        uint32_t c = cps[k];
        return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    };

    size_t i = 0;
    while (i < n) {
        size_t end = i;
        // 1. contractions
        if (cps[i] == '\'') {
            uint32_t a = lower(i + 1), b = lower(i + 2);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') end = i + 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) end = i + 3;
        }
        // 2. optional non-letter/number/newline prefix, then letters
        if (end == i) {
            if (L(i)) {
                end = i + 1;
                while (L(end)) end++;
            } else if (!NL(i) && !N(i) && L(i + 1)) {
                end = i + 2;
                while (L(end)) end++;
            }
        }
        // 3. a single number
        if (end == i && N(i)) end = i + 1;
        // 4. optional space, then punctuation/symbols, then newlines
        if (end == i) {
            size_t j = i;
            if (cps[j] == ' ' && other(j + 1)) j++;
            if (other(j)) {
                while (other(j)) j++;
                while (NL(j)) j++;
                end = j;
            }
        }
        // 5-7. white space
        if (end == i && S(i)) {
            size_t run_end = i;
            while (S(run_end)) run_end++;
            size_t last_nl = n;
            for (size_t k = i; k < run_end; k++)
                if (NL(k)) last_nl = k;
            if (last_nl != n) end = last_nl + 1;                      // 5. up to the last newline
            else if (run_end == n || run_end - i == 1) end = run_end;  // 6/7. whole run
            else end = run_end - 1;                                    // 6. leave one space for the next word
        }
        if (end == i) end = i + 1;  // unreachable for valid input; never loop forever
        pieces.emplace_back(s.substr(offs[i], offs[end] - offs[i]));
        i = end;
    }
}

// --- BPE ----------------------------------------------------------------------

void Tokenizer::bpe(std::string_view piece, std::vector<int> &out) const {
    if (piece.size() == 1) {
        out.push_back(byte_id_[static_cast<unsigned char>(piece[0])]);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(std::string(piece));
        if (it != cache_.end()) {
            out.insert(out.end(), it->second.begin(), it->second.end());
            return;
        }
    }

    struct Sym {
        int id, prev, next, len;
    };
    const int n = static_cast<int>(piece.size());
    std::vector<Sym> sym(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) sym[i] = {byte_id_[static_cast<unsigned char>(piece[i])], i - 1, i + 1 < n ? i + 1 : -1, 1};

    struct Cand {
        int rank, pos, result;
        bool operator<(const Cand &o) const {  // min-heap on (rank, pos)
            return rank != o.rank ? rank > o.rank : pos > o.pos;
        }
    };
    std::priority_queue<Cand> heap;
    auto consider = [&](int pos) {
        if (pos < 0) return;
        int nx = sym[pos].next;
        if (nx < 0) return;
        auto it = merges_.find(pair_key(sym[pos].id, sym[nx].id));
        if (it != merges_.end()) heap.push({it->second.rank, pos, it->second.result});
    };
    for (int i = 0; i + 1 < n; i++) consider(i);

    while (!heap.empty()) {
        Cand c = heap.top();
        heap.pop();
        Sym &left = sym[c.pos];
        if (left.len == 0 || left.next < 0) continue;
        Sym &right = sym[left.next];
        auto it = merges_.find(pair_key(left.id, right.id));
        if (it == merges_.end() || it->second.result != c.result) continue;  // stale candidate
        left.id = c.result;
        left.len += right.len;
        right.len = 0;
        left.next = right.next;
        if (left.next >= 0) sym[left.next].prev = c.pos;
        consider(left.prev);
        consider(c.pos);
    }

    std::vector<int> ids;
    for (int i = 0; i >= 0; i = sym[i].next) ids.push_back(sym[i].id);
    out.insert(out.end(), ids.begin(), ids.end());

    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (cache_.size() >= kCacheLimit) cache_.clear();
    cache_.emplace(std::string(piece), std::move(ids));
}

void Tokenizer::encode_segment(std::string_view text, std::vector<int> &out) const {
    if (text.empty()) return;
    std::string normalized = unicode::nfc(text);
    std::vector<std::string_view> pieces;
    pre_tokenize(normalized, pieces);
    for (std::string_view p : pieces) bpe(p, out);
}

std::vector<int> Tokenizer::encode(std::string_view text, bool parse_special) const {
    std::vector<int> out;
    out.reserve(text.size() / 3 + 4);
    size_t seg_start = 0;
    if (parse_special && !added_.empty()) {
        for (size_t i = 0; i < text.size();) {
            const Added *match = nullptr;
            for (const auto &a : added_) {
                if (text.compare(i, a.text.size(), a.text) == 0) {
                    match = &a;
                    break;  // longest first
                }
            }
            if (!match) {
                i++;
                continue;
            }
            encode_segment(text.substr(seg_start, i - seg_start), out);
            out.push_back(match->id);
            i += match->text.size();
            seg_start = i;
        }
    }
    encode_segment(text.substr(seg_start), out);
    return out;
}

std::string Tokenizer::decode(std::span<const int> ids, bool skip_special) const {
    std::string bytes;
    for (int id : ids) {
        if (skip_special && is_special(id)) continue;
        bytes += token_bytes(id);
    }
    return unicode::to_valid_utf8(bytes);
}

std::string StreamDecoder::push(int id) {
    if (skip_special_ && tok_->is_special(id)) return {};
    pending_ += tok_->token_bytes(id);
    size_t done = unicode::complete_utf8_prefix(pending_);
    if (done == 0) return {};
    std::string out = unicode::to_valid_utf8(std::string_view(pending_).substr(0, done));
    pending_.erase(0, done);
    return out;
}

std::string StreamDecoder::flush() {
    std::string out = unicode::to_valid_utf8(pending_);
    pending_.clear();
    return out;
}

}  // namespace ember
