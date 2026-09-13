#include "emprise/tokenizer.hpp"
#include "json.hpp"
#include <pcre2.h>
#include <utf8proc.h>
#include <array>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace emprise {
namespace {
std::string utf8(int cp) {
    utf8proc_uint8_t out[4];
    auto n = utf8proc_encode_char(cp, out);
    return std::string(reinterpret_cast<char*>(out), n);
}
}
struct Tokenizer::Impl {
    std::vector<std::string> vocab, decoded;
    std::vector<std::pair<std::string, int>> special;
    std::unordered_map<std::string, int> ids, rank;
    std::array<std::string, 256> byte_encoder;
    pcre2_code* regex = nullptr;
    ~Impl() { if (regex) pcre2_code_free(regex); }
    void ordinary(const std::string& text, std::vector<int>& result) const {
        auto* match = pcre2_match_data_create_from_pattern(regex, nullptr);
        if (!match) throw std::bad_alloc();
        std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> guard(match, pcre2_match_data_free);
        size_t offset = 0;
        while (offset < text.size()) {
            auto rc = pcre2_match(regex, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), offset, PCRE2_ANCHORED, match, nullptr);
            if (rc < 0) throw std::runtime_error("Invalid UTF-8 or unsupported tokenization input");
            auto* span = pcre2_get_ovector_pointer(match);
            if (span[0] != offset || span[1] <= offset) throw std::runtime_error("Tokenizer made no progress");
            std::vector<std::string> pieces;
            for (auto i = offset; i < span[1]; ++i) pieces.push_back(byte_encoder[static_cast<uint8_t>(text[i])]);
            while (pieces.size() > 1) {
                int best_rank = std::numeric_limits<int>::max();
                size_t best = pieces.size();
                for (size_t i = 0; i + 1 < pieces.size(); ++i) {
                    auto found = rank.find(pieces[i] + ' ' + pieces[i+1]);
                    if (found != rank.end() && found->second < best_rank) { best_rank = found->second; best = i; }
                }
                if (best == pieces.size()) break;
                pieces[best] += pieces[best+1];
                pieces.erase(pieces.begin() + static_cast<ptrdiff_t>(best+1));
            }
            for (const auto& piece : pieces) {
                auto found = ids.find(piece);
                if (found == ids.end()) throw std::runtime_error("BPE piece missing from vocabulary");
                result.push_back(found->second);
            }
            offset = span[1];
        }
    }
};

Tokenizer::Tokenizer(const std::string& metadata): impl_(std::make_unique<Impl>()) {
    auto m = nlohmann::json::parse(metadata);
    if (m.value("tokenizer.ggml.model", "") != "gpt2" || m.value("tokenizer.ggml.pre", "") != "qwen35")
        throw std::runtime_error("Tokenizer currently supports qwen35 GPT-2 BPE only");
    impl_->vocab = m.at("tokenizer.ggml.tokens").get<std::vector<std::string>>();
    auto types = m.at("tokenizer.ggml.token_type").get<std::vector<int>>();
    if (types.size() != impl_->vocab.size()) throw std::runtime_error("Vocabulary/type size mismatch");
    auto merges = m.at("tokenizer.ggml.merges").get<std::vector<std::string>>();
    for (size_t i = 0; i < merges.size(); ++i) impl_->rank.emplace(merges[i], static_cast<int>(i));
    int extra = 256;
    std::unordered_map<int, uint8_t> reverse;
    for (int b = 0; b < 256; ++b) {
        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
        const auto cp = direct ? b : extra++;
        impl_->byte_encoder[b] = utf8(cp);
        reverse[cp] = static_cast<uint8_t>(b);
    }
    for (size_t i = 0; i < impl_->vocab.size(); ++i) {
        const auto& token = impl_->vocab[i];
        impl_->ids.emplace(token, static_cast<int>(i));
        if (types[i] == 3 || types[i] == 4) {
            impl_->special.emplace_back(token, static_cast<int>(i));
            impl_->decoded.push_back(token);
            continue;
        }
        std::string bytes;
        for (size_t pos = 0; pos < token.size();) {
            utf8proc_int32_t cp;
            auto n = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(token.data()+pos), token.size()-pos, &cp);
            if (n <= 0) throw std::runtime_error("Invalid vocabulary UTF-8");
            auto it = reverse.find(cp);
            if (it == reverse.end()) { bytes = token; break; } // unused vocabulary entries
            bytes.push_back(static_cast<char>(it->second));
            pos += static_cast<size_t>(n);
        }
        impl_->decoded.push_back(std::move(bytes));
    }
    const char* pattern = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    int error;
    PCRE2_SIZE at;
    impl_->regex = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern), PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP, &error, &at, nullptr);
    if (!impl_->regex) throw std::runtime_error("Could not compile tokenizer regex");
}
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
    // GGUF qwen35 tokenization preserves bytes; unlike some HF tokenizer.json
    // exports, its metadata does not specify an NFC normalization stage.
    const std::string& normalized = text;
    std::vector<int> result;
    size_t offset = 0;
    while (offset < normalized.size()) {
        size_t next = normalized.size(), length = 0;
        int id = -1;
        if (parse_special) for (const auto& [s, token] : impl_->special) {
            auto pos = normalized.find(s, offset);
            if (pos < next || (pos == next && s.size() > length)) { next = pos; length = s.size(); id = token; }
        }
        impl_->ordinary(normalized.substr(offset, next-offset), result);
        if (id < 0) break;
        result.push_back(id);
        offset = next + length;
    }
    return result;
}
std::string Tokenizer::decode(int token) const {
    if (token < 0 || static_cast<size_t>(token) >= impl_->decoded.size()) throw std::out_of_range("Invalid token ID");
    return impl_->decoded[token];
}
std::string Tokenizer::chat_prompt(const std::string& user) const {
    // Deliberately restricted text-only single-turn template; arbitrary Jinja is not executed.
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}
}
