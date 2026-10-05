// A small JSON DOM: parsing (config.json, tokenizer.json, safetensors
// headers, HTTP request bodies) and serialization (API responses).
//
// Objects keep their keys in document order. Lookup is linear, which is the
// right trade-off here: objects are either small (requests, configs) or only
// iterated (the 150k-entry tokenizer vocabulary).
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ember::json {

class Value;
using Array = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;

class Value {
public:
    enum class Type { null, boolean, number, string, array, object };

    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool b) : v_(b) {}
    Value(double d) : v_(Number{d, false, 0}) {}
    Value(int i) : v_(Number{static_cast<double>(i), true, i}) {}
    Value(int64_t i) : v_(Number{static_cast<double>(i), true, i}) {}
    Value(uint64_t i) : v_(Number{static_cast<double>(i), true, static_cast<int64_t>(i)}) {}
    Value(const char *s) : v_(std::string(s)) {}
    Value(std::string s) : v_(std::move(s)) {}
    Value(std::string_view s) : v_(std::string(s)) {}
    Value(Array a) : v_(std::move(a)) {}
    Value(Object o) : v_(std::move(o)) {}

    // Throws ember::Error with the line and column of the problem.
    static Value parse(std::string_view text);

    Type type() const { return static_cast<Type>(v_.index()); }
    bool is_null() const { return type() == Type::null; }
    bool is_bool() const { return type() == Type::boolean; }
    bool is_number() const { return type() == Type::number; }
    bool is_string() const { return type() == Type::string; }
    bool is_array() const { return type() == Type::array; }
    bool is_object() const { return type() == Type::object; }

    // Typed access; each throws ember::Error naming the expected type on mismatch.
    bool as_bool() const;
    double as_number() const;
    int64_t as_int() const;  // also fails for non-integral numbers
    const std::string &as_string() const;
    const Array &as_array() const;
    const Object &as_object() const;
    Array &as_array();
    Object &as_object();

    // Object access.
    const Value *find(std::string_view key) const;  // nullptr if absent (or not an object)
    const Value &operator[](std::string_view key) const;  // throws if absent
    bool contains(std::string_view key) const { return find(key) != nullptr; }
    Value &set(std::string_view key, Value value);  // turns null into an object

    // Convenience getters with defaults, for optional fields.
    double get_number(std::string_view key, double def) const;
    int64_t get_int(std::string_view key, int64_t def) const;
    bool get_bool(std::string_view key, bool def) const;
    std::string get_string(std::string_view key, std::string_view def) const;

    // Array access.
    size_t size() const;
    const Value &operator[](size_t i) const;
    void push(Value value);  // turns null into an array

    std::string dump() const;
    void dump_to(std::string &out) const;

private:
    struct Number {
        double d;
        bool is_int;
        int64_t i;
    };
    std::variant<std::monostate, bool, Number, std::string, Array, Object> v_;
};

// Appends s as a quoted JSON string (escaping quotes, backslashes and control characters).
void write_string(std::string &out, std::string_view s);

}  // namespace ember::json
