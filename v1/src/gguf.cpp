#include "emprise/gguf.hpp"
#include "json.hpp"
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace emprise {
namespace {
using json = nlohmann::json;
uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::runtime_error("GGUF dimension overflow");
    return a * b;
}
class Reader {
    std::ifstream& f;
    uint64_t size;
public:
    Reader(std::ifstream& stream, uint64_t bytes): f(stream), size(bytes) {}
    template<class T> T scalar() {
        static_assert(std::endian::native == std::endian::little, "Little-endian host required");
        T result{};
        if (!f.read(reinterpret_cast<char*>(&result), sizeof(T)))
            throw std::runtime_error("Truncated GGUF header");
        return result;
    }
    std::string string() {
        const auto n = scalar<uint64_t>();
        const auto pos = static_cast<uint64_t>(f.tellg());
        if (pos > size || n > size - pos || n > 64 * 1024 * 1024)
            throw std::runtime_error("Invalid GGUF string length: " + std::to_string(n) +
                " at " + std::to_string(pos) + " in " + std::to_string(size) + " bytes");
        std::string s(static_cast<size_t>(n), '\0');
        if (!f.read(s.data(), static_cast<std::streamsize>(n)))
            throw std::runtime_error("Truncated GGUF string");
        return s;
    }
    json value(uint32_t type, int depth = 0) {
        if (depth > 4) throw std::runtime_error("Excessive metadata nesting");
        switch (type) {
        case 0: return scalar<uint8_t>();
        case 1: return scalar<int8_t>();
        case 2: return scalar<uint16_t>();
        case 3: return scalar<int16_t>();
        case 4: return scalar<uint32_t>();
        case 5: return scalar<int32_t>();
        case 6: return scalar<float>();
        case 7: {
            auto b = scalar<uint8_t>();
            if (b > 1) throw std::runtime_error("Invalid boolean metadata");
            return b == 1;
        }
        case 8: return string();
        case 9: {
            auto subtype = scalar<uint32_t>();
            auto count = scalar<uint64_t>();
            if (count > size || count > 10000000) throw std::runtime_error("Invalid metadata array");
            auto a = json::array();
            for (uint64_t i = 0; i < count; ++i) a.push_back(value(subtype, depth + 1));
            return a;
        }
        case 10: return scalar<uint64_t>();
        case 11: return scalar<int64_t>();
        case 12: return scalar<double>();
        default: throw std::runtime_error("Unknown GGUF metadata type");
        }
    }
};
}

uint64_t encoded_bytes(WeightType type, uint64_t n) {
    switch (type) {
    case WeightType::f32: return checked_mul(n, 4);
    case WeightType::f16:
    case WeightType::bf16: return checked_mul(n, 2);
    case WeightType::q6_k:
        if (n % 256) throw std::runtime_error("Q6_K requires blocks of 256 elements");
        return checked_mul(n / 256, 210);
    default: throw std::runtime_error("Unsupported GGUF tensor encoding: " + std::to_string(static_cast<uint32_t>(type)));
    }
}

Gguf::Gguf(const std::filesystem::path& path): file_(path, std::ios::binary | std::ios::ate) {
    if (!file_) throw std::runtime_error("Cannot open model: " + path.string());
    // Query the opened stream: some Windows toolchains report the reparse-point
    // size (zero) for Hugging Face cache symlinks instead of the target size.
    const auto end = file_.tellg();
    if (end < 0) throw std::runtime_error("Cannot determine model size");
    file_bytes_ = static_cast<uint64_t>(end);
    file_.seekg(0);
    Reader r(file_, file_bytes_);
    if (r.scalar<uint32_t>() != 0x46554747) throw std::runtime_error("Not a GGUF file");
    const auto version = r.scalar<uint32_t>();
    if (version != 2 && version != 3) throw std::runtime_error("Unsupported GGUF version");
    const auto nt = r.scalar<uint64_t>(), nkv = r.scalar<uint64_t>();
    if (nt > 1000000 || nkv > 1000000) throw std::runtime_error("Excessive header entries");
    json metadata = json::object();
    for (uint64_t i = 0; i < nkv; ++i) {
        auto name = r.string();
        if (metadata.contains(name)) throw std::runtime_error("Duplicate metadata key");
        metadata[name] = r.value(r.scalar<uint32_t>());
    }
    const auto alignment = metadata.value("general.alignment", uint64_t(32));
    if (!alignment || alignment > 1024 * 1024 || (alignment & (alignment - 1)))
        throw std::runtime_error("Invalid GGUF alignment");
    metadata_json_ = metadata.dump();
    tensors_.reserve(static_cast<size_t>(nt));
    for (uint64_t i = 0; i < nt; ++i) {
        Tensor t;
        t.name = r.string();
        auto nd = r.scalar<uint32_t>();
        if (!nd || nd > 4) throw std::runtime_error("Unsupported tensor rank");
        t.elements = 1;
        for (uint32_t d = 0; d < nd; ++d) {
            const auto n = r.scalar<uint64_t>();
            if (!n) throw std::runtime_error("Zero tensor dimension");
            t.shape.push_back(n);
            t.elements = checked_mul(t.elements, n);
        }
        t.type = static_cast<WeightType>(r.scalar<uint32_t>());
        t.bytes = encoded_bytes(t.type, t.elements);
        if (t.type == WeightType::q6_k && t.shape[0] % 256)
            throw std::runtime_error("Q6_K row must contain whole blocks");
        t.offset = r.scalar<uint64_t>();
        if (t.offset % alignment) throw std::runtime_error("Misaligned tensor offset");
        if (!index_.emplace(t.name, tensors_.size()).second)
            throw std::runtime_error("Duplicate tensor name");
        tensors_.push_back(std::move(t));
    }
    const auto header_end = static_cast<uint64_t>(file_.tellg());
    data_offset_ = (header_end + alignment - 1) / alignment * alignment;
    if (data_offset_ > file_bytes_) throw std::runtime_error("Missing tensor data");
    std::vector<std::pair<uint64_t, uint64_t>> extents;
    for (const auto& t : tensors_) {
        if (t.offset > data_bytes() || t.bytes > data_bytes() - t.offset)
            throw std::runtime_error("Truncated tensor: " + t.name);
        extents.emplace_back(t.offset, t.offset + t.bytes);
    }
    std::sort(extents.begin(), extents.end());
    for (size_t i = 1; i < extents.size(); ++i)
        if (extents[i].first < extents[i-1].second)
            throw std::runtime_error("Overlapping tensor extents");
}

const Tensor& Gguf::tensor(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) throw std::runtime_error("Missing tensor: " + name);
    return tensors_[it->second];
}

void Gguf::configure_storage(DiskPolicy policy, uint64_t budget) {
    if (data_bytes() > budget) {
        if (policy == DiskPolicy::disabled)
            throw std::runtime_error("Weights exceed RAM budget and disk offloading is disabled");
        std::vector<std::byte>().swap(resident_);
        return;
    }
    if (resident()) return;
    resident_.resize(static_cast<size_t>(data_bytes()));
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(data_offset_));
    // Bounded read operations work with streams that limit individual read sizes.
    for (uint64_t offset = 0; offset < data_bytes();) {
        const auto count = std::min<uint64_t>(64 * 1024 * 1024, data_bytes() - offset);
        if (!file_.read(reinterpret_cast<char*>(resident_.data() + offset), static_cast<std::streamsize>(count))) {
            std::vector<std::byte>().swap(resident_);
            throw std::runtime_error("Failed to load resident weights");
        }
        bytes_read_ += count;
        offset += count;
    }
}

void Gguf::read(const Tensor& t, uint64_t offset, std::span<std::byte> out) {
    if (offset > t.bytes || out.size() > t.bytes - offset ||
        t.offset > data_bytes() || t.bytes > data_bytes() - t.offset)
        throw std::out_of_range("Weight read exceeds tensor extent");
    if (resident()) {
        std::memcpy(out.data(), resident_.data() + t.offset + offset, out.size());
    } else {
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(data_offset_ + t.offset + offset));
        if (!file_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size())))
            throw std::runtime_error("Weight read failed: " + t.name);
        bytes_read_ += out.size();
    }
}
}
