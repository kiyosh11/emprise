#include "emprise/gguf.hpp"
#include "emprise/backend.hpp"
#include <cstdlib>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace emprise;
void check(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
float round_positive_half(float value) {
    // Independent monotonic search through finite positive half values.
    unsigned lo=0,hi=0x7bff;
    while(lo<hi) {unsigned mid=(lo+hi)/2;if(half_to_float(static_cast<uint16_t>(mid))<value) lo=mid+1;else hi=mid;}
    if(!lo) return 0;
    float a=half_to_float(static_cast<uint16_t>(lo-1)),b=half_to_float(static_cast<uint16_t>(lo));
    return value-a<b-value || (value-a==b-value && ((lo-1)&1)==0)?a:b;
}
template<class F> void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    check(threw, "Expected invalid input to fail");
}
template<class T> void write(std::ofstream& f, T value) { f.write(reinterpret_cast<char*>(&value), sizeof(value)); }
void string(std::ofstream& f, const std::string& s) { write<uint64_t>(f, s.size()); f.write(s.data(), s.size()); }
void fixture(const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary);
    write<uint32_t>(f, 0x46554747); write<uint32_t>(f, 3);
    write<uint64_t>(f, 1); write<uint64_t>(f, 0);
    string(f, "matrix"); write<uint32_t>(f, 2);
    write<uint64_t>(f, 2); write<uint64_t>(f, 2);
    write<uint32_t>(f, 0); write<uint64_t>(f, 0);
    while (static_cast<uint64_t>(f.tellp()) % 32) f.put('\0');
    for (float x : {1.f, 2.f, -3.f, 4.f}) write(f, x);
}

int main(int argc, char** argv) {
    try {
        check(half_to_float(0x3c00) == 1.f, "half one");
        check(half_to_float(0xc000) == -2.f, "half negative");
        check(half_to_float(1) == std::ldexp(1.f, -24), "half subnormal");
        check(std::signbit(half_to_float(0x8000)), "half negative zero");
        check(std::isinf(half_to_float(0x7c00)), "half infinity");
        check(std::isnan(half_to_float(0x7e00)), "half NaN");
        rejects([] { encoded_bytes(WeightType::q6_k, 255); });
        rejects([] { encoded_bytes(WeightType::f32, std::numeric_limits<uint64_t>::max()); });

        std::array<uint8_t, 210> packed{};
        std::array<float, 256> expected{}, decoded{}, x{};
        // Independently encode varying signed six-bit values and sixteen scales.
        for (int i = 0; i < 16; ++i) packed[192+i] = static_cast<uint8_t>(i - 8);
        const uint16_t scale = 0x3800; // half 0.5
        std::memcpy(packed.data() + 208, &scale, 2);
        for (int i = 0; i < 256; ++i) {
            const auto q = (i * 13 + 7) % 64;
            const auto half = i / 128, local = i % 128, segment = local / 32, lane = local % 32;
            const auto lower = half * 64 + (segment % 2) * 32 + lane;
            packed[lower] |= static_cast<uint8_t>((q & 15) << (segment / 2 * 4));
            packed[128 + half * 32 + lane] |= static_cast<uint8_t>((q >> 4) << (segment * 2));
            expected[i] = .5f * (i / 16 - 8) * (q - 32);
            x[i] = static_cast<float>((i % 7) - 3);
        }
        auto bytes = std::as_bytes(std::span(packed));
        dequantize(WeightType::q6_k, bytes, decoded);
        check(expected == decoded, "Q6_K dequantization differs from encoded values");
        float actual = 0, expected_dot = 0;
        for (int i = 0; i < 256; ++i) expected_dot += expected[i] * x[i];
        matvec(WeightType::q6_k, bytes, x, std::span(&actual, 1));
        check(actual == expected_dot, "Q6_K matvec");
        rejects([&] { dequantize(WeightType::q6_k, bytes.first(209), decoded); });

        const auto path = std::filesystem::temp_directory_path() / "emprise-core-fixture.gguf";
        fixture(path);
        {
            Gguf g(path);
            check(g.tensors().size() == 1, "Tensor count");
            rejects([&] { g.configure_storage(DiskPolicy::disabled, 1); });
            g.configure_storage(DiskPolicy::automatic, 1);
            check(!g.resident(), "Low budget must stream");
            auto& t = g.tensor("matrix");
            std::array<std::byte, 16> disk{}, ram{};
            g.read(t, 0, disk);
            rejects([&] { g.read(t, 1, disk); });
            g.configure_storage(DiskPolicy::disabled, 16);
            check(g.resident(), "Sufficient budget must load resident weights");
            auto reads = g.bytes_read();
            g.read(t, 0, ram);
            check(ram == disk && reads == g.bytes_read(), "Resident reads must not touch disk");
            std::array<float, 2> in{2, 3}, out{};
            matvec(WeightType::f32, ram, in, out);
            check(out[0] == 8 && out[1] == 6, "FP32 matrix multiplication");
            if (std::getenv("EMPRISE_TEST_CUDA")) {
                auto gpu=cuda_backend();
                gpu->linear(g,t,in,out);
                check(out[0] == 8 && out[1] == 6, "CUDA FP32 matrix multiplication");
                std::cout << "Tested " << gpu->name() << '\n';
            }
        }
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
        rejects([&] { Gguf g(path); });
        std::filesystem::remove(path);
        if (argc > 1 && std::getenv("EMPRISE_TEST_CUDA")) {
            Gguf real(argv[1]);
            const auto& t=real.tensor("blk.0.ssm_alpha.weight");
            std::vector<float> x(static_cast<size_t>(t.shape[0])), a(static_cast<size_t>(t.shape[1])), b(a.size());
            for(size_t i=0;i<x.size();++i) x[i]=std::sin(float(i)*.01f);
            auto cpu=cpu_backend();auto gpu=cuda_backend();
            cpu->linear(real,t,x,a);gpu->linear(real,t,x,b);
            float max_error=0;
            for(size_t i=0;i<a.size();++i) {
                max_error=std::max(max_error,std::abs(a[i]-b[i]));
                check(std::abs(a[i]-b[i]) <= 1e-3f + 1e-4f*std::abs(a[i]), "Real Q6 CPU/CUDA mismatch");
            }
            std::cout << "Real Q6 matrix CPU/CUDA max absolute error: " << max_error << '\n';
            // Validate the integer-dot path against explicitly quantized CPU input.
            auto qx=x;
            for(size_t start=0;start<x.size();start+=32) {
                float scale=0;
                for(size_t j=0;j<32;++j) scale=std::max(scale,std::abs(x[start+j]));
                scale/=127;
                for(size_t j=0;j<32;++j) qx[start+j]=scale?std::round(x[start+j]/scale)*round_positive_half(scale):0;
            }
            auto fast=cuda_backend(0,true);
            cpu->linear(real,t,qx,a);fast->linear(real,t,x,b);
            for(size_t i=0;i<a.size();++i)
                check(std::abs(a[i]-b[i]) <= 1e-3f + 1e-4f*std::abs(a[i]), "Q6/Q8 CPU/CUDA mismatch");
            const auto& gate=real.tensor("blk.0.ffn_gate.weight");
            const auto& up=real.tensor("blk.0.ffn_up.weight");
            const auto& down=real.tensor("blk.0.ffn_down.weight");
            a.resize(static_cast<size_t>(down.shape[1]));b.resize(a.size());
            // Compare the fused operation with separate GPU operations; this
            // isolates device residency/SwiGLU from CPU/GPU matvec rounding.
            for(Backend* tested:{gpu.get(),fast.get()}) {
                tested->Backend::feed_forward(real,gate,up,down,x,a);
                tested->feed_forward(real,gate,up,down,x,b);
                double squared=0,energy=0;
                for(size_t i=0;i<a.size();++i) {squared+=double(a[i]-b[i])*(a[i]-b[i]);energy+=double(a[i])*a[i];}
                const double relative=std::sqrt(squared/std::max(energy,1e-30));
                std::cout<<"Fused feed-forward relative L2: "<<relative<<'\n';
                check(relative<0.002,"Fused feed-forward differs from separate operations");
            }
            auto limited=cuda_backend(1);
            cpu->feed_forward(real,gate,up,down,x,a);limited->feed_forward(real,gate,up,down,x,b);
            check(a==b,"Feed-forward CPU fallback mismatch");
            // Batched projection must match the same number of single-token ones.
            {
                const auto& bt=real.tensor("blk.0.ffn_down.weight");
                const int K=4;
                const size_t cols=static_cast<size_t>(bt.shape[0]), rows=static_cast<size_t>(bt.shape[1]);
                std::vector<float> xb(cols*K), yseq(rows*K), ybat(rows*K);
                for(size_t i=0;i<xb.size();++i) xb[i]=std::sin(float(i)*0.013f);
                for(int k=0;k<K;++k)
                    gpu->linear(real,bt,std::span(xb).subspan(static_cast<size_t>(k)*cols,cols),
                                std::span(yseq).subspan(static_cast<size_t>(k)*rows,rows));
                gpu->linear_batch(real,bt,xb,ybat,K);
                double err=0,en=0;
                for(size_t i=0;i<yseq.size();++i) { double d=double(yseq[i])-ybat[i]; err+=d*d; en+=double(yseq[i])*yseq[i]; }
                const double rel=std::sqrt(err/std::max(en,1e-30));
                std::cout<<"Batched vs sequential relative L2: "<<rel<<'\n';
                check(rel<1e-4,"Batched matmul mismatch");
            }
        }
        std::cout << "All core tests passed\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
