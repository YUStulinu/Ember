#include "model/unicode.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace ember::unicode {

namespace {

#include "model/unicode_tables.inc"

enum : uint8_t { kLetter = 1, kNumber = 2, kSpace = 4 };

template <size_t N>
bool in_ranges(const uint32_t (&ranges)[N][2], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < ranges[mid][0]) hi = mid;
        else if (cp > ranges[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

// Class flags for the Basic Multilingual Plane, built once: almost all text
// lives there, and a table lookup beats a binary search in the hot loop.
struct BmpClasses {
    std::array<uint8_t, 0x10000> flags{};
    BmpClasses() {
        for (uint32_t cp = 0; cp < 0x10000; cp++) {
            uint8_t f = 0;
            if (in_ranges(kLetterRanges, cp)) f |= kLetter;
            if (in_ranges(kNumberRanges, cp)) f |= kNumber;
            if (in_ranges(kWhiteSpaceRanges, cp)) f |= kSpace;
            flags[cp] = f;
        }
    }
};

const BmpClasses &bmp() {
    static const BmpClasses classes;
    return classes;
}

// Hangul syllables compose and decompose algorithmically.
constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
constexpr uint32_t LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;

void decompose(uint32_t cp, std::vector<uint32_t> &out) {
    if (cp >= SBase && cp < SBase + SCount) {
        uint32_t s = cp - SBase;
        out.push_back(LBase + s / NCount);
        out.push_back(VBase + (s % NCount) / TCount);
        if (s % TCount) out.push_back(TBase + s % TCount);
        return;
    }
    if (cp >= 0xC0) {  // nothing below U+00C0 decomposes
        size_t lo = 0, hi = std::size(kDecomp);
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if (kDecomp[mid][0] < cp) lo = mid + 1;
            else hi = mid;
        }
        if (lo < std::size(kDecomp) && kDecomp[lo][0] == cp) {
            out.insert(out.end(), kDecompPool + kDecomp[lo][1], kDecompPool + kDecomp[lo][1] + kDecomp[lo][2]);
            return;
        }
    }
    out.push_back(cp);
}

constexpr uint32_t kNoComposite = 0xFFFFFFFF;

uint32_t compose(uint32_t a, uint32_t b) {
    if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount)
        return SBase + ((a - LBase) * VCount + (b - VBase)) * TCount;
    if (a >= SBase && a < SBase + SCount && (a - SBase) % TCount == 0 && b > TBase && b < TBase + TCount)
        return a + (b - TBase);
    size_t lo = 0, hi = std::size(kCompose);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (kCompose[mid][0] < a || (kCompose[mid][0] == a && kCompose[mid][1] < b)) lo = mid + 1;
        else hi = mid;
    }
    if (lo < std::size(kCompose) && kCompose[lo][0] == a && kCompose[lo][1] == b) return kCompose[lo][2];
    return kNoComposite;
}

}  // namespace

const char *data_version() { return EMBER_UNICODE_VERSION; }

uint32_t next_cp(std::string_view s, size_t &i) {
    auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    unsigned char c = b(i);
    if (c < 0x80) {
        i++;
        return c;
    }
    int len;
    uint32_t cp, min;
    if ((c & 0xE0) == 0xC0) {
        len = 2;
        cp = c & 0x1F;
        min = 0x80;
    } else if ((c & 0xF0) == 0xE0) {
        len = 3;
        cp = c & 0x0F;
        min = 0x800;
    } else if ((c & 0xF8) == 0xF0) {
        len = 4;
        cp = c & 0x07;
        min = 0x10000;
    } else {
        i++;
        return kReplacement;
    }
    if (i + len > s.size()) {
        i++;
        return kReplacement;
    }
    for (int k = 1; k < len; k++) {
        unsigned char cc = b(i + k);
        if ((cc & 0xC0) != 0x80) {
            i++;
            return kReplacement;
        }
        cp = (cp << 6) | (cc & 0x3F);
    }
    if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        i++;
        return kReplacement;
    }
    i += len;
    return cp;
}

void append_utf8(std::string &out, uint32_t cp) {
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

bool is_letter(uint32_t cp) { return cp < 0x10000 ? (bmp().flags[cp] & kLetter) : in_ranges(kLetterRanges, cp); }
bool is_number(uint32_t cp) { return cp < 0x10000 ? (bmp().flags[cp] & kNumber) : in_ranges(kNumberRanges, cp); }
bool is_space(uint32_t cp) { return cp < 0x10000 ? (bmp().flags[cp] & kSpace) : false; }

int combining_class(uint32_t cp) {
    if (cp < 0x300) return 0;
    size_t lo = 0, hi = std::size(kCccRanges);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < kCccRanges[mid][0]) hi = mid;
        else if (cp > kCccRanges[mid][1]) lo = mid + 1;
        else return static_cast<int>(kCccRanges[mid][2]);
    }
    return 0;
}

std::string nfc(std::string_view s) {
    // Fast path: valid UTF-8 with nothing at or above U+0300 is already NFC
    // (no combining marks, and no decomposable characters other than precomposed
    // Latin-1 letters, which NFC keeps). This covers English and most Romanian.
    bool simple = true;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            i++;
            continue;
        }
        size_t j = i;
        uint32_t cp = next_cp(s, j);
        if (cp >= 0x300 || cp == kReplacement) {
            simple = false;
            break;
        }
        i = j;
    }
    if (simple) return std::string(s);

    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    for (size_t i = 0; i < s.size();) decompose(next_cp(s, i), cps);

    // Canonical ordering: stable-sort each run of non-starters by combining class.
    for (size_t i = 0; i < cps.size();) {
        if (combining_class(cps[i]) == 0) {
            i++;
            continue;
        }
        size_t j = i;
        while (j < cps.size() && combining_class(cps[j]) != 0) j++;
        std::stable_sort(cps.begin() + static_cast<ptrdiff_t>(i), cps.begin() + static_cast<ptrdiff_t>(j),
                         [](uint32_t a, uint32_t b) { return combining_class(a) < combining_class(b); });
        i = j;
    }

    // Canonical composition (the algorithm of UAX #15).
    if (!cps.empty()) {
        size_t starter_pos = 0;
        uint32_t starter = cps[0];
        int last_class = combining_class(starter) ? 256 : 0;
        size_t out_pos = 1;
        for (size_t i = 1; i < cps.size(); i++) {
            uint32_t ch = cps[i];
            int ch_class = combining_class(ch);
            uint32_t composite = compose(starter, ch);
            if (composite != kNoComposite && (last_class < ch_class || last_class == 0)) {
                cps[starter_pos] = composite;
                starter = composite;
            } else {
                if (ch_class == 0) {
                    starter_pos = out_pos;
                    starter = ch;
                }
                last_class = ch_class;
                cps[out_pos++] = ch;
            }
        }
        cps.resize(out_pos);
    }

    std::string out;
    out.reserve(s.size());
    for (uint32_t cp : cps) append_utf8(out, cp);
    return out;
}

std::string to_valid_utf8(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0, n = s.size();
    auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    while (i < n) {
        unsigned char c = b(i);
        if (c < 0x80) {
            out += static_cast<char>(c);
            i++;
            continue;
        }
        int need;
        unsigned char lo = 0x80, hi = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) {
            need = 1;
        } else if (c >= 0xE0 && c <= 0xEF) {
            need = 2;
            if (c == 0xE0) lo = 0xA0;
            if (c == 0xED) hi = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            need = 3;
            if (c == 0xF0) lo = 0x90;
            if (c == 0xF4) hi = 0x8F;
        } else {
            append_utf8(out, kReplacement);
            i++;
            continue;
        }
        size_t j = i + 1;
        bool ok = true;
        for (int k = 0; k < need; k++) {
            if (j >= n || b(j) < lo || b(j) > hi) {
                ok = false;
                break;
            }
            j++;
            lo = 0x80;
            hi = 0xBF;
        }
        if (ok) out.append(s.data() + i, j - i);
        else append_utf8(out, kReplacement);
        i = j;  // on failure, the offending byte is examined again
    }
    return out;
}

size_t complete_utf8_prefix(std::string_view s) {
    // Look back at most 3 bytes for the start of the last sequence.
    size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        unsigned char c = static_cast<unsigned char>(s[n - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte
        size_t need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return back >= need ? n : n - back;
    }
    return n;  // only continuation bytes: invalid, let it through
}

}  // namespace ember::unicode
