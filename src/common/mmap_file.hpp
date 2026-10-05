// Read-only memory-mapped files. Model weights are mapped, not read: the OS
// pages them in on demand and they cost no private memory.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ember {

class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string &path);  // throws ember::Error
    ~MappedFile();
    MappedFile(MappedFile &&other) noexcept;
    MappedFile &operator=(MappedFile &&other) noexcept;
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    const uint8_t *data() const { return data_; }
    size_t size() const { return size_; }
    const std::string &path() const { return path_; }

private:
    void close();
    std::string path_;
    const uint8_t *data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    void *file_ = nullptr;
    void *mapping_ = nullptr;
#endif
};

// Reads a whole (small) file into a string; throws ember::Error.
std::string read_file(const std::string &path);
bool file_exists(const std::string &path);

}  // namespace ember
