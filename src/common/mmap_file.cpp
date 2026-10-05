#include "common/mmap_file.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

#include "common/error.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ember {

namespace {
std::filesystem::path to_path(const std::string &utf8) {
    return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}
}  // namespace

MappedFile::MappedFile(const std::string &path) : path_(path) {
#ifdef _WIN32
    HANDLE f = CreateFileW(to_path(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) fail("cannot open {} (error {})", path, GetLastError());
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz)) {
        CloseHandle(f);
        fail("cannot get the size of {}", path);
    }
    size_ = static_cast<size_t>(sz.QuadPart);
    file_ = f;
    if (size_ == 0) return;
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) {
        close();
        fail("cannot map {} (error {})", path, GetLastError());
    }
    mapping_ = m;
    data_ = static_cast<const uint8_t *>(MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0));
    if (!data_) {
        close();
        fail("cannot map a view of {} (error {})", path, GetLastError());
    }
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open {}", path);
    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        fail("cannot stat {}", path);
    }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ > 0) {
        void *p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            ::close(fd);
            fail("cannot map {}", path);
        }
        data_ = static_cast<const uint8_t *>(p);
    }
    ::close(fd);  // the mapping keeps the file alive
#endif
}

void MappedFile::close() {
#ifdef _WIN32
    if (data_) UnmapViewOfFile(data_);
    if (mapping_) CloseHandle(static_cast<HANDLE>(mapping_));
    if (file_) CloseHandle(static_cast<HANDLE>(file_));
    mapping_ = file_ = nullptr;
#else
    if (data_) munmap(const_cast<uint8_t *>(data_), size_);
#endif
    data_ = nullptr;
    size_ = 0;
}

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile &&o) noexcept { *this = std::move(o); }

MappedFile &MappedFile::operator=(MappedFile &&o) noexcept {
    if (this != &o) {
        close();
        path_ = std::move(o.path_);
        data_ = o.data_;
        size_ = o.size_;
        o.data_ = nullptr;
        o.size_ = 0;
#ifdef _WIN32
        file_ = o.file_;
        mapping_ = o.mapping_;
        o.file_ = o.mapping_ = nullptr;
#endif
    }
    return *this;
}

std::string read_file(const std::string &path) {
    std::ifstream in(to_path(path), std::ios::binary);
    if (!in) fail("cannot open {}", path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool file_exists(const std::string &path) {
    std::error_code ec;
    return std::filesystem::exists(to_path(path), ec);
}

}  // namespace ember
