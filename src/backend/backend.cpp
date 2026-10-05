#include "backend/backend.hpp"

#include "common/error.hpp"

namespace ember {

std::unique_ptr<Backend> create_cpu_backend(const std::string &dir, const BackendOptions &opt);
#ifdef EMBER_WITH_CUDA
std::unique_ptr<Backend> create_cuda_backend(const std::string &dir, const BackendOptions &opt);
bool cuda_device_present();
#endif

const char *weight_format_name(WeightFormat f) {
    switch (f) {
        case WeightFormat::f16: return "f16";
        case WeightFormat::int8: return "int8";
        case WeightFormat::int4: return "int4";
    }
    return "?";
}

WeightFormat parse_weight_format(const std::string &s) {
    if (s == "f16" || s == "fp16") return WeightFormat::f16;
    if (s == "int8" || s == "q8") return WeightFormat::int8;
    if (s == "int4" || s == "q4") return WeightFormat::int4;
    fail("unknown weight format \"{}\" (use f16, int8 or int4)", s);
}

bool cuda_available() {
#ifdef EMBER_WITH_CUDA
    return cuda_device_present();
#else
    return false;
#endif
}

std::unique_ptr<Backend> create_backend(const std::string &dir, const BackendOptions &opt) {
    if (opt.device == "cpu") return create_cpu_backend(dir, opt);
    if (opt.device == "cuda") {
#ifdef EMBER_WITH_CUDA
        return create_cuda_backend(dir, opt);
#else
        fail("this build of Ember has no CUDA support; use --device cpu");
#endif
    }
    fail("unknown device \"{}\" (use cuda or cpu)", opt.device);
}

}  // namespace ember
