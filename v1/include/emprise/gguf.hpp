#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace emprise {
enum class WeightType : uint32_t { f32 = 0, f16 = 1, q6_k = 14, bf16 = 30 };
enum class DiskPolicy { disabled, automatic, enabled };

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape; // GGUF order: fastest-varying dimension first.
    WeightType type;
    uint64_t offset = 0, bytes = 0, elements = 0;
};

// Read-only model storage. Resident mode performs no weight reads after loading.
// Streaming mode uses caller-owned bounded slices rather than a full FP32 copy.
class Gguf {
public:
    explicit Gguf(const std::filesystem::path& path);
    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;
    const std::vector<Tensor>& tensors() const noexcept { return tensors_; }
    const Tensor& tensor(const std::string& name) const;
    const std::string& metadata_json() const noexcept { return metadata_json_; }
    uint64_t data_bytes() const noexcept { return file_bytes_ - data_offset_; }
    uint64_t bytes_read() const noexcept { return bytes_read_; }
    bool resident() const noexcept { return !resident_.empty(); }
    void configure_storage(DiskPolicy policy, uint64_t weight_ram_budget);
    void read(const Tensor& tensor, uint64_t relative_offset, std::span<std::byte> output);
private:
    std::ifstream file_;
    uint64_t file_bytes_ = 0, data_offset_ = 0, bytes_read_ = 0;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string, size_t> index_;
    std::string metadata_json_;
    std::vector<std::byte> resident_;
};

uint64_t encoded_bytes(WeightType type, uint64_t elements);
float half_to_float(uint16_t bits) noexcept;
void dequantize(WeightType type, std::span<const std::byte> input, std::span<float> output);
// One matrix tile in row-major order, with independent rows parallelized.
void matvec(WeightType type, std::span<const std::byte> weights,
            std::span<const float> x, std::span<float> y);
}
