#pragma once
#include <memory>
#include <string>
#include <vector>

namespace emprise {
class Tokenizer {
public:
    explicit Tokenizer(const std::string& gguf_metadata_json);
    ~Tokenizer();
    std::vector<int> encode(const std::string& text, bool parse_special = false) const;
    std::string decode(int token) const;
    std::string chat_prompt(const std::string& user) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
