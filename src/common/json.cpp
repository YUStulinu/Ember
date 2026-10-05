#include "common/json.hpp"

#include <charconv>
#include <cmath>
#include <cstring>

#include "common/error.hpp"

namespace ember::json {

namespace {

constexpr int kMaxDepth = 256;

class Parser {
public:
    explicit Parser(std::string_view s) : s_(s) {}

    Value parse_document() {
        skip_ws();
        Value v = parse_value(0);
        skip_ws();
        if (pos_ != s_.size()) error("unexpected text after the JSON value");
        return v;
    }

private:
    std::string_view s_;
    size_t pos_ = 0;

    [[noreturn]] void error(const char *what) const {
        size_t line = 1, col = 1;
        for (size_t i = 0; i < pos_ && i < s_.size(); i++) {
            if (s_[i] == '\n') {
                line++;
                col = 1;
            } else {
                col++;
            }
        }
        fail("invalid JSON at line {}, column {}: {}", line, col, what);
    }

    void skip_ws() {
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') pos_++;
            else break;
        }
    }

    bool consume(char c) {
        if (pos_ < s_.size() && s_[pos_] == c) {
            pos_++;
            return true;
        }
        return false;
    }

    void expect_literal(std::string_view lit) {
        if (s_.substr(pos_, lit.size()) != lit) error("invalid literal");
        pos_ += lit.size();
    }

    Value parse_value(int depth) {
        if (depth > kMaxDepth) error("nesting too deep");
        if (pos_ >= s_.size()) error("unexpected end of input");
        switch (s_[pos_]) {
            case '{': return parse_object(depth);
            case '[': return parse_array(depth);
            case '"': return Value(parse_string());
            case 't': expect_literal("true"); return Value(true);
            case 'f': expect_literal("false"); return Value(false);
            case 'n': expect_literal("null"); return Value(nullptr);
            default: return parse_number();
        }
    }

    Value parse_object(int depth) {
        pos_++;  // '{'
        Object obj;
        skip_ws();
        if (consume('}')) return Value(std::move(obj));
        for (;;) {
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != '"') error("expected a string key");
            std::string key = parse_string();
            skip_ws();
            if (!consume(':')) error("expected ':'");
            skip_ws();
            Value v = parse_value(depth + 1);
            obj.emplace_back(std::move(key), std::move(v));
            skip_ws();
            if (consume(',')) continue;
            if (consume('}')) return Value(std::move(obj));
            error("expected ',' or '}'");
        }
    }

    Value parse_array(int depth) {
        pos_++;  // '['
        Array arr;
        skip_ws();
        if (consume(']')) return Value(std::move(arr));
        for (;;) {
            skip_ws();
            arr.push_back(parse_value(depth + 1));
            skip_ws();
            if (consume(',')) continue;
            if (consume(']')) return Value(std::move(arr));
            error("expected ',' or ']'");
        }
    }

    unsigned parse_hex4() {
        if (pos_ + 4 > s_.size()) error("truncated \\u escape");
        unsigned v = 0;
        for (int i = 0; i < 4; i++) {
            char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
            else error("invalid hex digit in \\u escape");
        }
        return v;
    }

    static void append_utf8(std::string &out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    std::string parse_string() {
        pos_++;  // opening quote
        std::string out;
        for (;;) {
            // Copy the run of plain characters in one go: strings dominate large files.
            size_t start = pos_;
            while (pos_ < s_.size() && s_[pos_] != '"' && s_[pos_] != '\\' &&
                   static_cast<unsigned char>(s_[pos_]) >= 0x20)
                pos_++;
            out.append(s_.data() + start, pos_ - start);
            if (pos_ >= s_.size()) error("unterminated string");
            char c = s_[pos_++];
            if (c == '"') return out;
            if (c != '\\') error("control character in string");
            if (pos_ >= s_.size()) error("unterminated escape");
            char e = s_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = parse_hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        // A high surrogate must be followed by a low one.
                        if (pos_ + 2 <= s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                            pos_ += 2;
                            uint32_t lo = parse_hex4();
                            if (lo < 0xDC00 || lo > 0xDFFF) error("invalid surrogate pair");
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else {
                            cp = 0xFFFD;
                        }
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        cp = 0xFFFD;
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: error("invalid escape");
            }
        }
    }

    Value parse_number() {
        size_t start = pos_;
        if (consume('-')) {}
        if (pos_ >= s_.size() || !(s_[pos_] >= '0' && s_[pos_] <= '9')) error("invalid value");
        bool integral = true;
        while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') pos_++;
        if (pos_ < s_.size() && s_[pos_] == '.') {
            integral = false;
            pos_++;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') pos_++;
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            integral = false;
            pos_++;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) pos_++;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') pos_++;
        }
        const char *b = s_.data() + start, *e = s_.data() + pos_;
        if (integral) {
            int64_t i = 0;
            auto r = std::from_chars(b, e, i);
            if (r.ec == std::errc() && r.ptr == e) return Value(i);
        }
        double d = 0;
        auto r = std::from_chars(b, e, d);
        if (r.ec != std::errc() || r.ptr != e) error("invalid number");
        return Value(d);
    }
};

const char *type_name(Value::Type t) {
    switch (t) {
        case Value::Type::null: return "null";
        case Value::Type::boolean: return "a boolean";
        case Value::Type::number: return "a number";
        case Value::Type::string: return "a string";
        case Value::Type::array: return "an array";
        case Value::Type::object: return "an object";
    }
    return "?";
}

[[noreturn]] void type_error(Value::Type want, Value::Type got) {
    fail("JSON: expected {}, found {}", type_name(want), type_name(got));
}

}  // namespace

Value Value::parse(std::string_view text) { return Parser(text).parse_document(); }

bool Value::as_bool() const {
    if (!is_bool()) type_error(Type::boolean, type());
    return std::get<bool>(v_);
}

double Value::as_number() const {
    if (!is_number()) type_error(Type::number, type());
    return std::get<Number>(v_).d;
}

int64_t Value::as_int() const {
    if (!is_number()) type_error(Type::number, type());
    const Number &n = std::get<Number>(v_);
    if (n.is_int) return n.i;
    if (std::floor(n.d) == n.d && std::fabs(n.d) < 9.0e15) return static_cast<int64_t>(n.d);
    fail("JSON: expected an integer, found {}", n.d);
}

const std::string &Value::as_string() const {
    if (!is_string()) type_error(Type::string, type());
    return std::get<std::string>(v_);
}

const Array &Value::as_array() const {
    if (!is_array()) type_error(Type::array, type());
    return std::get<Array>(v_);
}

const Object &Value::as_object() const {
    if (!is_object()) type_error(Type::object, type());
    return std::get<Object>(v_);
}

Array &Value::as_array() {
    if (!is_array()) type_error(Type::array, type());
    return std::get<Array>(v_);
}

Object &Value::as_object() {
    if (!is_object()) type_error(Type::object, type());
    return std::get<Object>(v_);
}

const Value *Value::find(std::string_view key) const {
    if (!is_object()) return nullptr;
    for (const auto &[k, v] : std::get<Object>(v_))
        if (k == key) return &v;
    return nullptr;
}

const Value &Value::operator[](std::string_view key) const {
    const Value *v = find(key);
    if (!v) fail("JSON: missing field \"{}\"", key);
    return *v;
}

Value &Value::set(std::string_view key, Value value) {
    if (is_null()) v_ = Object{};
    Object &obj = as_object();
    for (auto &[k, v] : obj)
        if (k == key) {
            v = std::move(value);
            return v;
        }
    obj.emplace_back(std::string(key), std::move(value));
    return obj.back().second;
}

double Value::get_number(std::string_view key, double def) const {
    const Value *v = find(key);
    return v && !v->is_null() ? v->as_number() : def;
}

int64_t Value::get_int(std::string_view key, int64_t def) const {
    const Value *v = find(key);
    return v && !v->is_null() ? v->as_int() : def;
}

bool Value::get_bool(std::string_view key, bool def) const {
    const Value *v = find(key);
    return v && !v->is_null() ? v->as_bool() : def;
}

std::string Value::get_string(std::string_view key, std::string_view def) const {
    const Value *v = find(key);
    return v && !v->is_null() ? v->as_string() : std::string(def);
}

size_t Value::size() const {
    if (is_array()) return std::get<Array>(v_).size();
    if (is_object()) return std::get<Object>(v_).size();
    return 0;
}

const Value &Value::operator[](size_t i) const {
    const Array &a = as_array();
    if (i >= a.size()) fail("JSON: index {} out of range ({} elements)", i, a.size());
    return a[i];
}

void Value::push(Value value) {
    if (is_null()) v_ = Array{};
    as_array().push_back(std::move(value));
}

void write_string(std::string &out, std::string_view s) {
    static const char hex[] = "0123456789abcdef";
    out += '"';
    size_t run = 0;
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c >= 0x20 && c != '"' && c != '\\') continue;
        out.append(s.data() + run, i - run);
        run = i + 1;
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                out += "\\u00";
                out += hex[c >> 4];
                out += hex[c & 15];
        }
    }
    out.append(s.data() + run, s.size() - run);
    out += '"';
}

void Value::dump_to(std::string &out) const {
    switch (type()) {
        case Type::null: out += "null"; break;
        case Type::boolean: out += std::get<bool>(v_) ? "true" : "false"; break;
        case Type::number: {
            const Number &n = std::get<Number>(v_);
            char buf[32];
            std::to_chars_result r;
            if (n.is_int) r = std::to_chars(buf, buf + sizeof buf, n.i);
            else if (!std::isfinite(n.d)) {
                out += "null";  // JSON has no NaN/Inf
                break;
            } else r = std::to_chars(buf, buf + sizeof buf, n.d);
            out.append(buf, r.ptr);
            break;
        }
        case Type::string: write_string(out, std::get<std::string>(v_)); break;
        case Type::array: {
            out += '[';
            bool first = true;
            for (const Value &v : std::get<Array>(v_)) {
                if (!first) out += ',';
                first = false;
                v.dump_to(out);
            }
            out += ']';
            break;
        }
        case Type::object: {
            out += '{';
            bool first = true;
            for (const auto &[k, v] : std::get<Object>(v_)) {
                if (!first) out += ',';
                first = false;
                write_string(out, k);
                out += ':';
                v.dump_to(out);
            }
            out += '}';
            break;
        }
    }
}

std::string Value::dump() const {
    std::string out;
    dump_to(out);
    return out;
}

}  // namespace ember::json
