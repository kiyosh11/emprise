#include "emprise/gguf.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace emprise {
float half_to_float(uint16_t h) noexcept {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 31, frac = h & 1023;
    if (exp == 0) {
        const float x = std::ldexp(static_cast<float>(frac), -24);
        return sign ? -x : x;
    }
    const uint32_t bits = sign | (exp == 31 ? 0x7f800000 : (exp + 112) << 23) | (frac << 13);
    return std::bit_cast<float>(bits);
}

namespace {
uint16_t u16(const std::byte* p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
void q6_block(const std::byte* raw, float* out) {
    const auto* b = reinterpret_cast<const uint8_t*>(raw);
    const auto* scales = reinterpret_cast<const int8_t*>(b + 192);
    const float d = half_to_float(u16(raw + 208));
    for (int half = 0; half < 2; ++half) {
        for (int group = 0; group < 4; ++group) {
            for (int lane = 0; lane < 32; ++lane) {
                const int low_index = half * 64 + (group & 1) * 32 + lane;
                const int low = (b[low_index] >> (group >= 2 ? 4 : 0)) & 15;
                const int high = (b[128 + half * 32 + lane] >> (group * 2)) & 3;
                const int q = (low | (high << 4)) - 32;
                out[half * 128 + group * 32 + lane] = d * scales[half * 8 + group * 2 + lane / 16] * q;
            }
        }
    }
}
void q4_k_block(const std::byte* raw, float* out) {
    const auto* b = reinterpret_cast<const uint8_t*>(raw);
    const float d = half_to_float(u16(raw + 0));
    const float dmin = half_to_float(u16(raw + 2));
    const uint8_t* scales = b + 4;
    const uint8_t* q = b + 16;
    int is = 0;
    auto scale_min = [&](int j, uint8_t& sc, uint8_t& mn) {
        if (j < 4) { sc = scales[j] & 63; mn = scales[j + 4] & 63; }
        else { sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
               mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4); }
    };
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, mn;
        scale_min(is + 0, sc, mn); const float d1 = d * sc, m1 = dmin * mn;
        scale_min(is + 1, sc, mn); const float d2 = d * sc, m2 = dmin * mn;
        for (int l = 0; l < 32; ++l) out[j + l] = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) out[j + 32 + l] = d2 * (q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}
}

void dequantize(WeightType type, std::span<const std::byte> in, std::span<float> out) {
    if (in.size() != encoded_bytes(type, out.size())) throw std::invalid_argument("Incorrect encoded size");
    if (type == WeightType::f32) {
        std::memcpy(out.data(), in.data(), in.size());
    } else if (type == WeightType::q6_k) {
        for (size_t i = 0; i < out.size() / 256; ++i) q6_block(in.data() + i * 210, out.data() + i * 256);
    } else if (type == WeightType::q4_k) {
        for (size_t i = 0; i < out.size() / 256; ++i) q4_k_block(in.data() + i * 144, out.data() + i * 256);
    } else {
        for (size_t i = 0; i < out.size(); ++i) {
            auto bits = u16(in.data() + i * 2);
            out[i] = type == WeightType::bf16 ? std::bit_cast<float>(uint32_t(bits) << 16) : half_to_float(bits);
        }
    }
}

void matvec(WeightType type, std::span<const std::byte> w, std::span<const float> x, std::span<float> y) {
    if (x.empty()) throw std::invalid_argument("Empty input vector");
    const auto row_bytes = encoded_bytes(type, x.size());
    if (w.size() / row_bytes != y.size() || w.size() % row_bytes)
        throw std::invalid_argument("Matrix size mismatch");
    // FP32 accumulation with original quantized weight values; no activation quantization.
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < static_cast<int64_t>(y.size()); ++row) {
        const auto* ptr = w.data() + row * row_bytes;
        float sum = 0;
        if (type == WeightType::q6_k || type == WeightType::q4_k) {
            const size_t block_bytes = type == WeightType::q6_k ? 210 : 144;
            float block[256];
            for (size_t k = 0; k < x.size(); k += 256) {
                if (type == WeightType::q6_k) q6_block(ptr + k / 256 * block_bytes, block);
                else q4_k_block(ptr + k / 256 * block_bytes, block);
                for (size_t j = 0; j < 256; ++j) sum += block[j] * x[k+j];
            }
        } else {
            for (size_t k = 0; k < x.size(); ++k) {
                float v;
                if (type == WeightType::f32) std::memcpy(&v, ptr + k * 4, 4);
                else {
                    auto bits = u16(ptr + k * 2);
                    v = type == WeightType::bf16 ? std::bit_cast<float>(uint32_t(bits) << 16) : half_to_float(bits);
                }
                sum += v * x[k];
            }
        }
        y[row] = sum;
    }
}
}
