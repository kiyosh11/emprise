#pragma once
#include "backend.hpp"
#include "tokenizer.hpp"
#include <atomic>
#include <functional>

namespace emprise {
struct ModelOptions {
    size_t context = 4096;
    uint64_t weight_ram_budget = 0; // zero: choose 70% of currently available RAM
    DiskPolicy disk = DiskPolicy::automatic;
    bool cuda = true;
    uint64_t weight_vram_budget = 0;
    bool quantize_activations = false;
    bool fused_ffn = true;
    // CPU worker threads; zero selects a conservative default from the core count.
    unsigned threads = 0;
    // Optional synchronous diagnostic hook; the span is valid only during the call.
    std::function<void(size_t, const std::string&, std::span<const float>)> trace;
};
struct Profile {
    double matrix_seconds=0, non_matrix_seconds=0;
    double norm_seconds=0, attention_seconds=0, recurrent_seconds=0;
    uint64_t matrix_calls=0, evaluated_tokens=0;
};
class Model {
public:
    Model(const std::filesystem::path& path, ModelOptions options);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    const Tokenizer& tokenizer() const;
    std::string backend_name() const;
    Profile profile() const;
    void reset();
    std::vector<float> evaluate(int token, bool output_logits = true);
    // Processes a block of tokens together (batched projections) and returns the
    // logits for each position. Used for prompt prefill and draft verification.
    std::vector<std::vector<float>> evaluate_block(const std::vector<int>& tokens);
    // Processes the prompt in batched chunks and returns the final position's logits.
    std::vector<float> prefill(const std::vector<int>& tokens);
    // MTP (blk.32 next-n) draft head. Given the previous hidden and token,
    // returns logits for the following token.
    std::vector<float> last_hidden() const;
    std::vector<float> last_hidden_raw() const;
    std::vector<float> mtp_draft(const std::vector<float>& hidden, int token, size_t position);
    std::vector<float> mtp_hidden() const;
    void reset_mtp();
    // Callback receives committed greedy tokens; returning false cancels.
    void generate(const std::vector<int>& prompt, size_t count,
                  const std::function<bool(int)>& output, const std::atomic_bool* cancelled = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
