// Reading .safetensors model files: an 8-byte header length, a JSON header
// describing every tensor, then the raw little-endian tensor data. Files are
// memory-mapped, so a TensorView points straight into the file.
//
// A model split into shards (model-00001-of-00002.safetensors, ...) is opened
// through model.safetensors.index.json.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/mmap_file.hpp"

namespace ember {

enum class DType { f32, f16, bf16, i8, u8, i32, i64 };

const char *dtype_name(DType t);
size_t dtype_size(DType t);

struct TensorView {
    std::string name;
    DType dtype = DType::f32;
    std::vector<int64_t> shape;
    const void *data = nullptr;
    size_t bytes = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
    int64_t dim(size_t i) const { return shape.at(i); }
    std::string shape_str() const;
};

class SafeTensors {
public:
    // Opens model.safetensors or, if present, model.safetensors.index.json in `model_dir`.
    static SafeTensors open_dir(const std::string &model_dir);
    // Opens a single file.
    static SafeTensors open_file(const std::string &path);

    bool has(const std::string &name) const { return tensors_.count(name) != 0; }
    const TensorView &get(const std::string &name) const;  // throws if missing
    // Like get(), and also checks the shape.
    const TensorView &get(const std::string &name, const std::vector<int64_t> &shape) const;
    const std::map<std::string, TensorView> &all() const { return tensors_; }
    size_t total_bytes() const;

private:
    void add_file(const std::string &path);
    std::vector<std::unique_ptr<MappedFile>> files_;
    std::map<std::string, TensorView> tensors_;
};

}  // namespace ember
