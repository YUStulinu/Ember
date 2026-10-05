// Scalar conversions between float and the 16-bit formats in model files.
// (Vectorized versions live in the CPU kernels; these are for setup and tests.)
#pragma once

#include <cstdint>
#include <cstring>

namespace ember {

inline float bf16_to_float(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t float_to_bf16(float f) {  // round to nearest even
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x7FFFFFu)) return static_cast<uint16_t>((bits >> 16) | 0x40);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

inline float fp16_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal: normalize
            int e = -1;
            do {
                e++;
                man <<= 1;
            } while (!(man & 0x400u));
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((man & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t float_to_fp16(float f) {  // round to nearest even, overflow to inf
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t absx = x & 0x7FFFFFFFu;
    if (absx >= 0x7F800000u) return static_cast<uint16_t>(sign | 0x7C00u | (absx > 0x7F800000u ? 0x200u : 0));
    if (absx >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);  // rounds above 65504
    if (absx < 0x38800000u) {  // subnormal or zero in fp16
        if (absx < 0x33000000u) return static_cast<uint16_t>(sign);
        uint32_t man = (absx & 0x7FFFFFu) | 0x800000u;
        int shift = 113 - static_cast<int>(absx >> 23) + 13;
        uint32_t half_man = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_man & 1))) half_man++;
        return static_cast<uint16_t>(sign | half_man);
    }
    uint32_t v = absx - 0x38000000u;  // rebias exponent 127 -> 15
    uint32_t rounded = (v + 0xFFFu + ((v >> 13) & 1u)) >> 13;
    return static_cast<uint16_t>(sign | rounded);
}

}  // namespace ember
