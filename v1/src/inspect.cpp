#include "emprise/gguf.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: emprise-inspect MODEL.gguf\n";
        return 2;
    }
    try {
        emprise::Gguf model(argv[1]);
        std::cout << "Validated tensors: " << model.tensors().size()
                  << "\nEncoded data bytes: " << model.data_bytes() << '\n';
        for (const auto& t : model.tensors()) {
            std::cout << t.name << " type=" << static_cast<uint32_t>(t.type) << " shape=";
            for (auto n : t.shape) std::cout << n << ' ';
            std::cout << "bytes=" << t.bytes << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
