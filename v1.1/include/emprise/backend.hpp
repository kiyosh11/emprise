#pragma once
#include "gguf.hpp"
#include <memory>

namespace emprise {
// One projection in a group that shares a single input activation.
struct LinearRequest {
    const Tensor* tensor = nullptr;
    std::span<float> output;
};
class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual void linear(Gguf& model, const Tensor& tensor, std::span<const float> x, std::span<float> y) = 0;
    // Runs several projections that share one input vector. The CUDA backend
    // uploads the input once and launches the kernels back-to-back; the default
    // implementation just calls linear() for each request.
    virtual void linear_many(Gguf& model, std::span<const LinearRequest> requests, std::span<const float> x);
    // Batched projection: y_k = W * x_k for k in [0, K). x holds K contiguous
    // rows of tensor.shape[0] floats, y holds K contiguous rows of
    // tensor.shape[1] floats. Reuses each weight read across K tokens.
    virtual void linear_batch(Gguf& model, const Tensor& tensor, std::span<const float> x,
                              std::span<float> y, int K);
    // Computes down(silu(gate(x)) * up(x)); the CUDA backend can retain the
    // intermediate activations on-device. Default implementation supports fallback.
    virtual void feed_forward(Gguf& model, const Tensor& gate, const Tensor& up,
                              const Tensor& down, std::span<const float> x, std::span<float> y);
    // The non-projection math of one gated-delta-net (recurrent) layer: causal
    // convolution, query/key scaling, the state scan, and the per-head
    // normalization + gate. conv_state and rec_state persist across tokens.
    struct DeltaNet {
        int key_heads = 0, value_heads = 0, state_dim = 0, conv_width = 0;
        float eps = 0;
        int slot = 0;
        std::span<const float> qkv, gate, alpha, beta, conv, ssm_a, dt, norm;
        std::span<float> conv_state, rec_state, out;
    };
    virtual void delta_net(Gguf& model, const DeltaNet& params);
    // One complete recurrent (gated-delta-net) layer. The CUDA backend keeps the
    // projections, the scan, and the output projection on-device so only the
    // final result crosses the bus.
    struct RecurrentBlock {
        const Tensor *qkv_w = nullptr, *gate_w = nullptr, *alpha_w = nullptr, *beta_w = nullptr, *out_w = nullptr;
        std::span<const float> conv, ssm_a, dt, norm;
        int key_heads = 0, value_heads = 0, state_dim = 0, conv_width = 0;
        float eps = 0;
        int slot = 0;
        std::span<const float> x;
        std::span<float> conv_state, rec_state, out;
    };
    virtual void recurrent_block(Gguf& model, const RecurrentBlock& params);
    // Clears any backend-owned recurrent state (called on model reset).
    virtual void reset_state() {}
    // Optional device-resident path: the CUDA backend overrides these so the
    // residual stream can stay on the GPU for a whole token. The model only
    // calls them when device_path() is true.
    virtual bool device_path() const { return false; }
    virtual uint64_t device_capacity() const { return 0; }
    virtual uint64_t dev_alloc(size_t bytes) { (void)bytes; return 0; }
    virtual void dev_free(uint64_t) {}
    virtual void dev_upload(uint64_t, const void*, size_t) {}
    virtual void dev_download(void*, uint64_t, size_t) {}
    virtual void dev_rmsnorm(Gguf&, uint64_t, const Tensor&, uint64_t, int, float) {}
    virtual void dev_add(uint64_t, uint64_t, int) {}
    virtual void dev_linear(Gguf&, const Tensor&, uint64_t, uint64_t) {}
    virtual void dev_ffn(Gguf&, const Tensor&, const Tensor&, const Tensor&, uint64_t, uint64_t) {}
    virtual void dev_recurrent(Gguf&, const RecurrentBlock&, uint64_t, uint64_t) {}
};
std::unique_ptr<Backend> cpu_backend();
// Throws a diagnostic if CUDA/NVRTC is unavailable. A zero budget uses current
// free GPU memory minus a reserve; callers may explicitly select CPU instead.
std::unique_ptr<Backend> cuda_backend(uint64_t weight_budget = 0, bool quantize_activations = false);
}
