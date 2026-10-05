#include "model/safetensors.hpp"

#include <cstring>
#include <format>
#include <set>

#include "common/error.hpp"
#include "common/json.hpp"

namespace ember {

const char *dtype_name(DType t) {
    switch (t) {
        case DType::f32: return "F32";
        case DType::f16: return "F16";
        case DType::bf16: return "BF16";
        case DType::i8: return "I8";
        case DType::u8: return "U8";
        case DType::i32: return "I32";
        case DType::i64: return "I64";
    }
    return "?";
}

size_t dtype_size(DType t) {
    switch (t) {
        case DType::f32: case DType::i32: return 4;
        case DType::f16: case DType::bf16: return 2;
        case DType::i8: case DType::u8: return 1;
        case DType::i64: return 8;
    }
    return 0;
}

namespace {
DType parse_dtype(const std::string &s, const std::string &file) {
    if (s == "F32") return DType::f32;
    if (s == "F16") return DType::f16;
    if (s == "BF16") return DType::bf16;
    if (s == "I8") return DType::i8;
    if (s == "U8") return DType::u8;
    if (s == "I32") return DType::i32;
    if (s == "I64") return DType::i64;
    fail("{}: unsupported tensor dtype {}", file, s);
}
}  // namespace

std::string TensorView::shape_str() const {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) s += std::format("{}{}", i ? ", " : "", shape[i]);
    return s + "]";
}

void SafeTensors::add_file(const std::string &path) {
    auto file = std::make_unique<MappedFile>(path);
    const uint8_t *p = file->data();
    size_t size = file->size();
    if (size < 8) fail("{}: too small to be a safetensors file", path);
    uint64_t header_len = 0;
    for (int i = 0; i < 8; i++) header_len |= static_cast<uint64_t>(p[i]) << (8 * i);
    if (header_len > size - 8 || header_len > (100u << 20)) fail("{}: invalid header length", path);
    json::Value header = json::Value::parse(std::string_view(reinterpret_cast<const char *>(p + 8), header_len));
    const uint8_t *data = p + 8 + header_len;
    size_t data_size = size - 8 - header_len;

    for (const auto &[name, info] : header.as_object()) {
        if (name == "__metadata__") continue;
        TensorView t;
        t.name = name;
        t.dtype = parse_dtype(info["dtype"].as_string(), path);
        for (const auto &d : info["shape"].as_array()) t.shape.push_back(d.as_int());
        const auto &off = info["data_offsets"];
        auto begin = static_cast<uint64_t>(off[0].as_int()), end = static_cast<uint64_t>(off[1].as_int());
        if (begin > end || end > data_size) fail("{}: tensor {} lies outside the file", path, name);
        t.bytes = end - begin;
        if (t.bytes != static_cast<uint64_t>(t.numel()) * dtype_size(t.dtype))
            fail("{}: tensor {} has {} bytes for shape {}", path, name, t.bytes, t.shape_str());
        t.data = data + begin;
        if (!tensors_.emplace(name, std::move(t)).second) fail("tensor {} appears in more than one file", name);
    }
    files_.push_back(std::move(file));
}

SafeTensors SafeTensors::open_file(const std::string &path) {
    SafeTensors st;
    st.add_file(path);
    return st;
}

SafeTensors SafeTensors::open_dir(const std::string &dir) {
    SafeTensors st;
    std::string index = dir + "/model.safetensors.index.json";
    if (file_exists(index)) {
        json::Value idx = json::Value::parse(read_file(index));
        std::set<std::string> shards;
        for (const auto &[tensor, file] : idx["weight_map"].as_object()) shards.insert(file.as_string());
        for (const auto &f : shards) st.add_file(dir + "/" + f);
    } else {
        st.add_file(dir + "/model.safetensors");
    }
    return st;
}

const TensorView &SafeTensors::get(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) fail("the model has no tensor named {}", name);
    return it->second;
}

const TensorView &SafeTensors::get(const std::string &name, const std::vector<int64_t> &shape) const {
    const TensorView &t = get(name);
    if (t.shape != shape) {
        TensorView want;
        want.shape = shape;
        fail("tensor {} has shape {}, expected {}", name, t.shape_str(), want.shape_str());
    }
    return t;
}

size_t SafeTensors::total_bytes() const {
    size_t n = 0;
    for (const auto &[_, t] : tensors_) n += t.bytes;
    return n;
}

}  // namespace ember
