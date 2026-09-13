// Diagnostic: times device matrix operations on real tensors to separate kernel
// throughput from per-call transfer/synchronization overhead.
#include "emprise/gguf.hpp"
#include "emprise/backend.hpp"
#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace emprise;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Usage: kernel-bench MODEL.gguf [repetitions] [fast]");
        const int repetitions = argc > 2 ? std::stoi(argv[2]) : 50;
        const bool fast = argc > 3 && std::string(argv[3]) == "fast";
        Gguf model(argv[1]);
        auto backend = fast ? cuda_backend(0, true) : cuda_backend();
        auto has = [&](const std::string& name) {
            for (const auto& t : model.tensors()) if (t.name == name) return true;
            return false;
        };
        std::cout << "Backend: " << backend->name() << '\n';
        for (const std::string& name : {"blk.0.attn_qkv.weight", "blk.0.ffn_gate.weight",
                                        "blk.0.ffn_down.weight", "blk.1.attn_qkv.weight",
                                        "output.weight", "token_embd.weight"}) {
            if (!has(name)) continue;
            const auto& t = model.tensor(name);
            if (t.shape.size() != 2) continue;
            std::vector<float> x(static_cast<size_t>(t.shape[0])), y(static_cast<size_t>(t.shape[1]));
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.01f);
            auto upload_start = Clock::now();
            backend->linear(model, t, x, y);
            const double upload = std::chrono::duration<double>(Clock::now() - upload_start).count();
            auto start = Clock::now();
            for (int r = 0; r < repetitions; ++r) backend->linear(model, t, x, y);
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            const double gib = double(t.bytes) / (1024.0 * 1024 * 1024);
            std::cout.precision(4);
            std::cout << name << " bytes=" << t.bytes
                      << " first_call=" << upload << "s"
                      << " steady_ms=" << seconds / repetitions * 1000
                      << " GiB/s=" << gib * repetitions / seconds;
            // Batched: K tokens in one weight pass.
            const int K = 4;
            std::vector<float> xb(x.size() * K), yb(y.size() * K);
            for (int k = 0; k < K; ++k) std::copy(x.begin(), x.end(), xb.begin() + k * x.size());
            backend->linear_batch(model, t, xb, yb, K); // warm
            auto bstart = Clock::now();
            for (int r = 0; r < repetitions; ++r) backend->linear_batch(model, t, xb, yb, K);
            const double bseconds = std::chrono::duration<double>(Clock::now() - bstart).count();
            std::cout << " batch4_ms=" << bseconds / repetitions * 1000
                      << " batch4_tok/s=" << K * repetitions / bseconds * 1000
                      << " amortization=" << (seconds / repetitions) / (bseconds / repetitions) * K << "x\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
